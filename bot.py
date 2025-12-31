import os
import re
import html
import json
import asyncio
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from datetime import datetime

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from aiolimiter import AsyncLimiter

from telegram import Update, InputMediaPhoto
from telegram.constants import ParseMode
from telegram.error import BadRequest, RetryAfter, TimedOut, NetworkError, Forbidden
from telegram.request import HTTPXRequest
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

# =========================
# ENV + LOG (CLEAN)
# =========================
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# Riduce lo spam dei log HTTP
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
log = logging.getLogger("cc-top")

BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_USER_ID = os.getenv("OWNER_USER_ID")
OUTPUT_CHAT_ID = os.getenv("OUTPUT_CHAT_ID")

MULEBUY_REF = os.getenv("MULEBUY_REF", "200836051")
CNFANS_REF = os.getenv("CNFANS_REF", "222394")

TZ = os.getenv("TZ", "Europe/Rome")
MAX_PHOTOS = int(os.getenv("MAX_PHOTOS", "4"))

# Album: attesa per raccogliere tutte le foto nello stesso media_group
MEDIA_GROUP_WAIT = float(os.getenv("MEDIA_GROUP_WAIT", "0.6"))

# Coda: puoi mandare 50+ post di fila senza blocchi
QUEUE_MAX = int(os.getenv("QUEUE_MAX", "2000"))

# Limite invio verso Telegram (anti rate-limit / anti impallo):
# esempi: 1.2 msg/sec circa (molto stabile)
SEND_RATE = float(os.getenv("SEND_RATE", "1.2"))  # "invii" al secondo
SEND_BURST = int(os.getenv("SEND_BURST", "2"))    # burst massimo

# Timeout rete più larghi (Railway)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))

STATE_FILE = os.getenv("STATE_FILE", "state.json")  # su Railway può essere volatile, ok

if not BOT_TOKEN:
    raise RuntimeError("Missing BOT_TOKEN")

OWNER_USER_ID_INT = int(OWNER_USER_ID) if OWNER_USER_ID and OWNER_USER_ID.strip() else None
OUTPUT_CHAT_ID_INT = int(OUTPUT_CHAT_ID) if OUTPUT_CHAT_ID and OUTPUT_CHAT_ID.strip() else None

# =========================
# STATE: daily header
# =========================
def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: dict) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

STATE = load_state()

def today_str() -> str:
    return datetime.now(ZoneInfo(TZ)).strftime("%d/%m/%Y")

async def send_daily_header_if_needed(bot) -> None:
    if not OUTPUT_CHAT_ID_INT:
        return
    key = f"last_header:{OUTPUT_CHAT_ID_INT}"
    t = today_str()
    if STATE.get(key) == t:
        return
    await bot.send_message(
        chat_id=OUTPUT_CHAT_ID_INT,
        text=f"📅 <b>{t}</b>",
        parse_mode=ParseMode.HTML,
    )
    STATE[key] = t
    save_state(STATE)

# =========================
# Robust Retry Wrapper
# =========================
async def tg_retry(coro_factory, retries: int = 7, base_sleep: float = 0.8):
    """
    Retry robusto su:
    - RetryAfter (429)
    - TimedOut / NetworkError (problemi rete)
    """
    for attempt in range(retries):
        try:
            return await coro_factory()
        except RetryAfter as e:
            wait = float(getattr(e, "retry_after", 3))
            await asyncio.sleep(wait)
        except (TimedOut, NetworkError) as e:
            await asyncio.sleep(base_sleep * (2 ** attempt))
    return await coro_factory()

# =========================
# URL rewrite (solo mulebuy/cnfans)
# =========================
URL_RE = re.compile(r"(https?://[^\s<>\]]+)", re.IGNORECASE)
HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)
A_TAG_RE = re.compile(r'<a\s+href="([^"]+)">(.+?)</a>', re.IGNORECASE | re.DOTALL)
TRAILING_PUNCT = ".,;:!?)\u201d\u2019]"

def strip_trailing_punct(url: str):
    trail = ""
    while url and url[-1] in TRAILING_PUNCT:
        trail = url[-1] + trail
        url = url[:-1]
    return url, trail

def normalize_netloc(netloc: str) -> str:
    return netloc.lower().lstrip("www.")

def rebuild_url(p, new_query: str) -> str:
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))

def transform_mulebuy(url: str) -> str:
    p = urlparse(url)
    if normalize_netloc(p.netloc) != "mulebuy.com":
        return url
    if not p.path.startswith("/product"):
        return url
    q = list(parse_qsl(p.query, keep_blank_values=True))
    q = [(k, v) for (k, v) in q if k.lower() != "ref"]
    q.append(("ref", MULEBUY_REF))
    return rebuild_url(p, urlencode(q, doseq=True))

def transform_cnfans(url: str) -> str:
    p = urlparse(url)
    if normalize_netloc(p.netloc) != "cnfans.com":
        return url
    if not p.path.startswith("/product"):
        return url

    q = list(parse_qsl(p.query, keep_blank_values=True))
    q = [(k, v) for (k, v) in q if k.lower() != "ref"]

    platform = None
    pid = None
    others = []
    for k, v in q:
        kl = k.lower()
        if kl == "platform" and platform is None:
            platform = v
        elif kl == "id" and pid is None:
            pid = v
        else:
            others.append((k, v))

    ordered = []
    if platform is not None:
        ordered.append(("platform", platform))
    if pid is not None:
        ordered.append(("id", pid))
    ordered.append(("ref", CNFANS_REF))
    ordered.extend(others)
    return rebuild_url(p, urlencode(ordered, doseq=True))

def transform_url(url: str) -> str:
    out = transform_mulebuy(url)
    out = transform_cnfans(out)
    return out

def rewrite_html_message_safe(html_text: str) -> tuple[str, bool]:
    """
    Riscrive href e url visibili senza rompere l'HTML Telegram.
    """
    changed = False

    def href_repl(m):
        nonlocal changed
        old = html.unescape(m.group(1))
        new = transform_url(old)
        if new != old:
            changed = True
        return f'href="{html.escape(new, quote=True)}"'

    out = HREF_RE.sub(href_repl, html_text)

    parts = re.split(r"(<[^>]+>)", out)

    def repl_visible(m):
        nonlocal changed
        raw = m.group(1)
        base, trail = strip_trailing_punct(raw)
        new = transform_url(base)
        if new != base:
            changed = True
        return html.escape(new, quote=False) + trail

    for i in range(0, len(parts), 2):  # solo testo
        parts[i] = URL_RE.sub(repl_visible, parts[i])

    return "".join(parts), changed

def html_to_plain_with_links(html_text: str, max_len: int = 1024) -> str:
    if not html_text:
        return ""
    t = re.sub(r"<br\s*/?>", "\n", html_text, flags=re.IGNORECASE)

    def a_repl(m):
        url = html.unescape(m.group(1)).strip()
        label = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        label = html.unescape(label)
        return f"{label}: {url}" if label else url

    t = A_TAG_RE.sub(a_repl, t)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    if len(t) > max_len:
        t = t[: max_len - 1] + "…"
    return t

# =========================
# SEND QUEUE (sequential, stable)
# =========================
SEND_QUEUE: asyncio.Queue[dict] = asyncio.Queue(maxsize=QUEUE_MAX)
SENDER_TASK: Optional[asyncio.Task] = None

# limiter: max ~SEND_RATE invii/s con burst SEND_BURST
SEND_LIMITER = AsyncLimiter(max_rate=SEND_BURST, time_period=1.0 / max(SEND_RATE, 0.01))

async def enqueue(job: dict):
    """
    Non droppa: se la coda è piena, aspetta (così 50 post non spariscono).
    """
    await SEND_QUEUE.put(job)

async def sender_loop(app):
    bot = app.bot
    while True:
        job = await SEND_QUEUE.get()
        try:
            if not OUTPUT_CHAT_ID_INT:
                continue

            # 1) data del giorno (1 sola volta)
            async with SEND_LIMITER:
                await tg_retry(lambda: send_daily_header_if_needed(bot))

            kind = job.get("kind")

            if kind == "album":
                photos: List[str] = job.get("photos", [])
                cap_html: Optional[str] = job.get("caption_html")
                cap_plain: Optional[str] = job.get("caption_plain")

                # (A) Prova caption HTML (se fallisce entity parsing -> fallback)
                try:
                    media = []
                    for i, fid in enumerate(photos):
                        if i == 0 and cap_html:
                            media.append(InputMediaPhoto(media=fid, caption=cap_html, parse_mode=ParseMode.HTML))
                        else:
                            media.append(InputMediaPhoto(media=fid))
                    async with SEND_LIMITER:
                        await tg_retry(lambda: bot.send_media_group(chat_id=OUTPUT_CHAT_ID_INT, media=media))
                except BadRequest:
                    # (B) fallback plain
                    try:
                        media2 = []
                        for i, fid in enumerate(photos):
                            if i == 0 and cap_plain:
                                media2.append(InputMediaPhoto(media=fid, caption=cap_plain))
                            else:
                                media2.append(InputMediaPhoto(media=fid))
                        async with SEND_LIMITER:
                            await tg_retry(lambda: bot.send_media_group(chat_id=OUTPUT_CHAT_ID_INT, media=media2))
                    except Exception:
                        # (C) last resort: no caption
                        media3 = [InputMediaPhoto(media=fid) for fid in photos]
                        async with SEND_LIMITER:
                            await tg_retry(lambda: bot.send_media_group(chat_id=OUTPUT_CHAT_ID_INT, media=media3))

            elif kind == "photo":
                fid = job["photo"]
                cap_html = job.get("caption_html")
                cap_plain = job.get("caption_plain")

                try:
                    async with SEND_LIMITER:
                        await tg_retry(lambda: bot.send_photo(
                            chat_id=OUTPUT_CHAT_ID_INT,
                            photo=fid,
                            caption=(cap_html if cap_html else None),
                            parse_mode=(ParseMode.HTML if cap_html else None),
                        ))
                except BadRequest:
                    try:
                        async with SEND_LIMITER:
                            await tg_retry(lambda: bot.send_photo(
                                chat_id=OUTPUT_CHAT_ID_INT,
                                photo=fid,
                                caption=(cap_plain if cap_plain else None),
                            ))
                    except Exception:
                        async with SEND_LIMITER:
                            await tg_retry(lambda: bot.send_photo(chat_id=OUTPUT_CHAT_ID_INT, photo=fid))

            elif kind == "text":
                html_txt = job["text_html"]
                plain_txt = job["text_plain"]
                try:
                    async with SEND_LIMITER:
                        await tg_retry(lambda: bot.send_message(
                            chat_id=OUTPUT_CHAT_ID_INT,
                            text=html_txt,
                            parse_mode=ParseMode.HTML,
                            disable_web_page_preview=True,
                        ))
                except BadRequest:
                    async with SEND_LIMITER:
                        await tg_retry(lambda: bot.send_message(
                            chat_id=OUTPUT_CHAT_ID_INT,
                            text=plain_txt,
                            disable_web_page_preview=True,
                        ))

        except Forbidden:
            # bot non ha permessi nel gruppo
            log.error("Forbidden: il bot non ha permessi nel gruppo. Rendilo admin.")
        except Exception as e:
            # mai "Task exception was never retrieved"
            log.exception("Sender error: %s", e)
        finally:
            SEND_QUEUE.task_done()

async def post_init(app):
    global SENDER_TASK
    if SENDER_TASK is None or SENDER_TASK.done():
        SENDER_TASK = asyncio.create_task(sender_loop(app), name="sender-loop")
        log.info("Sender loop started.")

# =========================
# ALBUM BUFFER (gathers media_group)
# =========================
@dataclass
class MediaGroupBuffer:
    media_group_id: str
    photo_file_ids: List[str] = field(default_factory=list)
    caption_html: Optional[str] = None
    task: Optional[asyncio.Task] = None

BUFFERS: Dict[str, MediaGroupBuffer] = {}
LOCK = asyncio.Lock()

async def finalize_media_group(context: ContextTypes.DEFAULT_TYPE, key: str):
    try:
        await asyncio.sleep(MEDIA_GROUP_WAIT)
        async with LOCK:
            buf = BUFFERS.pop(key, None)

        if not buf:
            return

        photos = list(dict.fromkeys(buf.photo_file_ids))
        if len(photos) > MAX_PHOTOS:
            photos = random.sample(photos, k=MAX_PHOTOS)

        if not photos:
            return

        cap_html = None
        cap_plain = None
        if buf.caption_html:
            cap_html, _ = rewrite_html_message_safe(buf.caption_html)
            cap_plain = html_to_plain_with_links(cap_html)

        await enqueue({
            "kind": "album",
            "photos": photos,
            "caption_html": cap_html,
            "caption_plain": cap_plain,
        })
    except Exception as e:
        log.exception("finalize_media_group failed: %s", e)

# =========================
# COMMANDS
# =========================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Bot TOP pronto (CLEAN + QUEUE).\n"
        "Puoi inoltrare 50 post di fila: li metto in coda e li pubblico in ordine.",
        disable_web_page_preview=True,
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if OWNER_USER_ID_INT and update.effective_user.id != OWNER_USER_ID_INT:
        return
    await update.message.reply_text(
        f"📦 Coda: {SEND_QUEUE.qsize()} / {QUEUE_MAX}\n"
        f"📅 Oggi: {today_str()}",
        disable_web_page_preview=True,
    )

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"🆔 Chat ID: <code>{update.effective_chat.id}</code>\n"
        f"🆔 Your User ID: <code>{update.effective_user.id}</code>",
        parse_mode=ParseMode.HTML,
    )

# =========================
# MAIN HANDLER (DM only + owner only)
# =========================
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    # solo chat privata
    if update.effective_chat.type != "private":
        return

    # owner only
    if OWNER_USER_ID_INT and update.effective_user.id != OWNER_USER_ID_INT:
        return

    if not OUTPUT_CHAT_ID_INT:
        return

    # Album: raccogli solo foto (video ignorati)
    if msg.media_group_id:
        key = str(msg.media_group_id)
        async with LOCK:
            buf = BUFFERS.get(key)
            if not buf:
                buf = MediaGroupBuffer(media_group_id=key)
                BUFFERS[key] = buf

            if msg.photo:
                buf.photo_file_ids.append(msg.photo[-1].file_id)

            if not buf.caption_html and msg.caption_html:
                buf.caption_html = msg.caption_html

            # reset timer: aspetta finché arrivano altre foto dello stesso album
            if buf.task and not buf.task.done():
                buf.task.cancel()
            buf.task = asyncio.create_task(finalize_media_group(context, key))
        return

    # Video singolo: ignora (clean)
    if msg.video:
        return

    # Foto singola
    if msg.photo:
        cap_html = None
        cap_plain = None
        if msg.caption_html:
            cap_html, _ = rewrite_html_message_safe(msg.caption_html)
            cap_plain = html_to_plain_with_links(cap_html)
        await enqueue({
            "kind": "photo",
            "photo": msg.photo[-1].file_id,
            "caption_html": cap_html,
            "caption_plain": cap_plain,
        })
        return

    # Testo
    if msg.text_html:
        rewritten_html, changed = rewrite_html_message_safe(msg.text_html)
        if not changed:
            return  # clean: non pubblichiamo se non c'è conversione
        plain = html_to_plain_with_links(rewritten_html, max_len=4096)
        await enqueue({"kind": "text", "text_html": rewritten_html, "text_plain": plain})
        return

# =========================
# MAIN
# =========================
def main():
    request = HTTPXRequest(
        connect_timeout=HTTP_TIMEOUT,
        read_timeout=HTTP_TIMEOUT,
        write_timeout=HTTP_TIMEOUT,
        pool_timeout=HTTP_TIMEOUT,
    )

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .request(request)
        .concurrent_updates(64)     # intake veloce (tu mandi 50 post)
        .post_init(post_init)       # avvia sender loop
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle))

    log.info("Bot running (TOP CLEAN + QUEUE)...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )

if __name__ == "__main__":
    main()
