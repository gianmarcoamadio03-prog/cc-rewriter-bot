import os
import re
import html
import json
import asyncio
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from datetime import datetime

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from telegram import Update, InputMediaPhoto
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.request import HTTPXRequest
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

# =====================================
# ENV + LOG
# =====================================
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("cc-rewriter-clean")

BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_USER_ID = os.getenv("OWNER_USER_ID")
OUTPUT_CHAT_ID = os.getenv("OUTPUT_CHAT_ID")

MULEBUY_REF = os.getenv("MULEBUY_REF", "200836051")
CNFANS_REF = os.getenv("CNFANS_REF", "222394")

TZ = os.getenv("TZ", "Europe/Rome")
MAX_PHOTOS = int(os.getenv("MAX_PHOTOS", "4"))
MEDIA_GROUP_WAIT = float(os.getenv("MEDIA_GROUP_WAIT", "0.9"))

STATE_FILE = os.getenv("STATE_FILE", "state.json")  # volatile su Railway, ok

if not BOT_TOKEN:
    raise RuntimeError("Missing BOT_TOKEN")

OWNER_USER_ID_INT = int(OWNER_USER_ID) if OWNER_USER_ID and OWNER_USER_ID.strip() else None
OUTPUT_CHAT_ID_INT = int(OUTPUT_CHAT_ID) if OUTPUT_CHAT_ID and OUTPUT_CHAT_ID.strip() else None

# =====================================
# DAILY HEADER STATE
# =====================================
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

async def send_daily_header_if_needed(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not OUTPUT_CHAT_ID_INT:
        return
    key = f"last_header:{OUTPUT_CHAT_ID_INT}"
    t = today_str()
    if STATE.get(key) == t:
        return
    await context.bot.send_message(
        chat_id=OUTPUT_CHAT_ID_INT,
        text=f"📅 <b>{t}</b>",
        parse_mode=ParseMode.HTML,
    )
    STATE[key] = t
    save_state(STATE)

# =====================================
# URL REWRITE (ONLY Mulebuy / CNfans)
# =====================================
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

def rewrite_plain_text(text: str) -> tuple[str, bool]:
    changed = False

    def repl(m):
        nonlocal changed
        raw = m.group(1)
        base, trail = strip_trailing_punct(raw)
        new = transform_url(base)
        if new != base:
            changed = True
        return new + trail

    return URL_RE.sub(repl, text), changed

def rewrite_html_message_safe(html_text: str) -> tuple[str, bool]:
    """
    Riscrive in modo sicuro:
    - href="..." nei tag <a>
    - eventuali URL nel testo fuori dai tag
    Evita di generare HTML malformato.
    """
    changed = False

    # 1) riscrivi href="..."
    def href_repl(m):
        nonlocal changed
        old = html.unescape(m.group(1))
        new = transform_url(old)
        if new != old:
            changed = True
        return f'href="{html.escape(new, quote=True)}"'

    out = HREF_RE.sub(href_repl, html_text)

    # 2) riscrivi URL visibili SOLO fuori dai tag (<...>)
    parts = re.split(r"(<[^>]+>)", out)  # tag in indici dispari

    def repl_visible(m):
        nonlocal changed
        raw = m.group(1)
        base, trail = strip_trailing_punct(raw)
        new = transform_url(base)
        if new != base:
            changed = True
        return html.escape(new, quote=False) + trail

    for i in range(0, len(parts), 2):
        parts[i] = URL_RE.sub(repl_visible, parts[i])

    return "".join(parts), changed

def html_to_plain_with_links(html_text: str, max_len: int = 1024) -> str:
    """
    Converte HTML Telegram in testo:
    - trasforma <a href="URL">LABEL</a> in "LABEL: URL"
    - preserva i link (importante!)
    - rimuove altri tag
    """
    if not html_text:
        return ""

    # <br> -> newline
    t = re.sub(r"<br\s*/?>", "\n", html_text, flags=re.IGNORECASE)

    # <a href="url">label</a> -> label: url
    def a_repl(m):
        url = html.unescape(m.group(1)).strip()
        label = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        label = html.unescape(label)
        if label:
            return f"{label}: {url}"
        return url

    t = A_TAG_RE.sub(a_repl, t)

    # rimuovi altri tag
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()

    # limita lunghezza caption telegram
    if len(t) > max_len:
        t = t[: max_len - 1] + "…"
    return t

# =====================================
# MEDIA GROUP BUFFER (albums)
# =====================================
@dataclass
class MediaGroupBuffer:
    media_group_id: str
    photo_file_ids: List[str] = field(default_factory=list)
    caption_html: Optional[str] = None
    task: Optional[asyncio.Task] = None

BUFFERS: Dict[str, MediaGroupBuffer] = {}
LOCK = asyncio.Lock()

async def finalize_media_group(context: ContextTypes.DEFAULT_TYPE, key: str):
    await asyncio.sleep(MEDIA_GROUP_WAIT)

    async with LOCK:
        buf = BUFFERS.pop(key, None)

    if not buf or not OUTPUT_CHAT_ID_INT:
        return

    await send_daily_header_if_needed(context)

    caption_html = None
    caption_plain = None
    if buf.caption_html:
        caption_html, _ = rewrite_html_message_safe(buf.caption_html)
        caption_plain = html_to_plain_with_links(caption_html)

    photos = list(dict.fromkeys(buf.photo_file_ids))
    if len(photos) > MAX_PHOTOS:
        photos = random.sample(photos, k=MAX_PHOTOS)

    if not photos:
        # niente foto: non pubblichiamo nulla (clean)
        return

    # 1) prova con caption HTML (se c'è)
    try:
        media = []
        for i, fid in enumerate(photos):
            if i == 0 and caption_html:
                media.append(InputMediaPhoto(media=fid, caption=caption_html, parse_mode=ParseMode.HTML))
            else:
                media.append(InputMediaPhoto(media=fid))
        await context.bot.send_media_group(chat_id=OUTPUT_CHAT_ID_INT, media=media)
        return
    except BadRequest as e:
        msg = str(e).lower()
        # se HTML non valido -> fallback testo plain (sempre clean)
        if "can't parse entities" not in msg and "can't parse inputmedia" not in msg:
            log.warning("send_media_group failed (non-entity): %s", e)

    # 2) fallback: caption PLAIN (senza parse_mode)
    try:
        media2 = []
        for i, fid in enumerate(photos):
            if i == 0 and caption_plain:
                media2.append(InputMediaPhoto(media=fid, caption=caption_plain))
            else:
                media2.append(InputMediaPhoto(media=fid))
        await context.bot.send_media_group(chat_id=OUTPUT_CHAT_ID_INT, media=media2)
        return
    except Exception as e:
        log.warning("send_media_group fallback plain failed: %s", e)

    # 3) ultimo fallback: album senza caption (pur di non sporcare)
    try:
        media3 = [InputMediaPhoto(media=fid) for fid in photos]
        await context.bot.send_media_group(chat_id=OUTPUT_CHAT_ID_INT, media=media3)
    except Exception as e:
        log.exception("Failed to send album: %s", e)

# =====================================
# COMMANDS
# =====================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Bot pronto (modalità CLEAN).\n"
        "Inoltra i post in privato: io pubblico nel gruppo solo data + post convertito.",
        disable_web_page_preview=True,
    )

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"🆔 Chat ID: <code>{update.effective_chat.id}</code>\n"
        f"🆔 Your User ID: <code>{update.effective_user.id}</code>",
        parse_mode=ParseMode.HTML,
    )

# =====================================
# MAIN HANDLER (DM only + owner only)
# =====================================
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    # solo chat privata
    if update.effective_chat.type != "private":
        return

    # se OWNER_USER_ID è settato, accetta solo te
    if OWNER_USER_ID_INT and update.effective_user.id != OWNER_USER_ID_INT:
        return

    if not OUTPUT_CHAT_ID_INT:
        await msg.reply_text("⚠️ OUTPUT_CHAT_ID non impostato (Railway Variables).")
        return

    # ALBUM: raccogli solo foto (video ignorati)
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

            if buf.task and not buf.task.done():
                buf.task.cancel()
            buf.task = asyncio.create_task(finalize_media_group(context, key))
        return

    # VIDEO singolo: IGNORA (clean)
    if msg.video:
        return

    # FOTO singola
    if msg.photo:
        await send_daily_header_if_needed(context)

        caption_html = None
        caption_plain = None
        if msg.caption_html:
            caption_html, _ = rewrite_html_message_safe(msg.caption_html)
            caption_plain = html_to_plain_with_links(caption_html)

        # prova HTML
        try:
            await context.bot.send_photo(
                chat_id=OUTPUT_CHAT_ID_INT,
                photo=msg.photo[-1].file_id,
                caption=(caption_html if caption_html else None),
                parse_mode=(ParseMode.HTML if caption_html else None),
            )
            return
        except BadRequest:
            pass

        # fallback plain
        try:
            await context.bot.send_photo(
                chat_id=OUTPUT_CHAT_ID_INT,
                photo=msg.photo[-1].file_id,
                caption=(caption_plain if caption_plain else None),
            )
            return
        except Exception:
            pass

        # ultimo fallback: senza caption
        await context.bot.send_photo(
            chat_id=OUTPUT_CHAT_ID_INT,
            photo=msg.photo[-1].file_id,
        )
        return

    # TESTO
    if msg.text_html:
        # riscrivi e invia SOLO se c'è una conversione (clean)
        rewritten_html, changed = rewrite_html_message_safe(msg.text_html)
        if not changed:
            return

        await send_daily_header_if_needed(context)

        # prova HTML
        try:
            await context.bot.send_message(
                chat_id=OUTPUT_CHAT_ID_INT,
                text=rewritten_html,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            return
        except BadRequest:
            pass

        # fallback plain
        plain = html_to_plain_with_links(rewritten_html, max_len=4096)
        await context.bot.send_message(
            chat_id=OUTPUT_CHAT_ID_INT,
            text=plain,
            disable_web_page_preview=True,
        )

def main():
    # Hardening timeout rete (evita timeout random)
    request = HTTPXRequest(connect_timeout=30, read_timeout=30, write_timeout=30, pool_timeout=30)

    app = ApplicationBuilder().token(BOT_TOKEN).request(request).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle))

    log.info("Bot running (CLEAN)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
