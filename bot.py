import os
import re
import html
import json
import asyncio
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
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
# ENV + LOG
# =========================
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
log = logging.getLogger("cc-top")


def must_env(name: str) -> str:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        raise RuntimeError(f"Missing env var: {name}")
    return str(v).strip()


BOT_TOKEN = must_env("BOT_TOKEN")
OWNER_USER_ID_INT = int(must_env("OWNER_USER_ID"))
OUTPUT_CHAT_ID_INT = int(must_env("OUTPUT_CHAT_ID"))

TZ = os.getenv("TZ", "Europe/Rome")
ZONE = ZoneInfo(TZ)

# affiliate refs
MULEBUY_REF = os.getenv("MULEBUY_REF", "200836051")
CNFANS_REF = os.getenv("CNFANS_REF", "222394")

# perf / behavior
MAX_PHOTOS = int(os.getenv("MAX_PHOTOS", "4"))                 # max foto per post (album)
MEDIA_GROUP_WAIT = float(os.getenv("MEDIA_GROUP_WAIT", "0.7")) # debounce album
MAX_ALBUM_AGE = float(os.getenv("MAX_ALBUM_AGE", "6.0"))       # watchdog album
QUEUE_MAX = int(os.getenv("QUEUE_MAX", "2000"))

# rates (separati: testo vs media)
TEXT_RATE = float(os.getenv("TEXT_RATE", "12"))   # msg testo/sec
MEDIA_RATE = float(os.getenv("MEDIA_RATE", "5"))  # media/sec

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))

# batch log
BATCH_WAIT = float(os.getenv("BATCH_WAIT", "3.5"))          # fallback timer
PAIR_WINDOW = float(os.getenv("PAIR_WINDOW", "10.0"))
BATCH_MIN_MESSAGES = int(os.getenv("BATCH_MIN_MESSAGES", "1"))

# caption/text safety
ALBUM_CAPTION_LIMIT = int(os.getenv("ALBUM_CAPTION_LIMIT", "950"))
PHOTO_CAPTION_LIMIT = int(os.getenv("PHOTO_CAPTION_LIMIT", "950"))
TEXT_LIMIT = int(os.getenv("TEXT_LIMIT", "3900"))

# ✅ una sola conferma per raffica (stessa sorgente) dopo idle
FAST_FINALIZE_IDLE = float(os.getenv("FAST_FINALIZE_IDLE", "1.5"))

STATE_FILE = os.getenv("STATE_FILE", "state.json")

# =========================
# STATE (persistente)
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
    return datetime.now(ZONE).strftime("%d/%m/%Y")


async def send_daily_header_if_needed(bot) -> None:
    if os.getenv("DAILY_HEADER", "0") != "1":
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
async def tg_retry(coro_factory, retries: int = 8, base_sleep: float = 0.8):
    for attempt in range(retries):
        try:
            return await coro_factory()
        except RetryAfter as e:
            wait = float(getattr(e, "retry_after", 3))
            await asyncio.sleep(wait + 0.2)
        except (TimedOut, NetworkError):
            await asyncio.sleep(base_sleep * (2 ** attempt))
    return await coro_factory()

# =========================
# URL rewrite (FORZA SOLO REF)
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
    # gestisce www. e m.
    n = (netloc or "").lower()
    if n.startswith("www."):
        n = n[4:]
    if n.startswith("m."):
        n = n[2:]
    return n


def rebuild_url(p, new_query: str) -> str:
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))


def transform_mulebuy(url: str) -> str:
    p = urlparse(url)
    if normalize_netloc(p.netloc) != "mulebuy.com":
        return url

    # gestisce ref doppi: parse come lista
    qlist = list(parse_qsl(p.query, keep_blank_values=True))
    # rimuovi TUTTI i ref esistenti
    qlist = [(k, v) for (k, v) in qlist if k.lower() != "ref"]
    # aggiungi SEMPRE il tuo
    qlist.append(("ref", MULEBUY_REF))

    return rebuild_url(p, urlencode(qlist, doseq=True))


def transform_cnfans(url: str) -> str:
    p = urlparse(url)
    if normalize_netloc(p.netloc) != "cnfans.com":
        return url

    qlist = list(parse_qsl(p.query, keep_blank_values=True))
    qlist = [(k, v) for (k, v) in qlist if k.lower() != "ref"]
    qlist.append(("ref", CNFANS_REF))

    return rebuild_url(p, urlencode(qlist, doseq=True))


def transform_url(url: str) -> str:
    out = transform_mulebuy(url)
    out = transform_cnfans(out)
    return out


def rewrite_html_message_safe(html_text: str) -> tuple[str, bool]:
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


def html_to_plain_with_links(html_text: str, max_len: int = 2048) -> str:
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


def truncate_text(s: str, limit: int) -> str:
    if not s:
        return ""
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"

# =========================
# Priority SEND QUEUE
# =========================
SEND_QUEUE: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=QUEUE_MAX)
SENDER_TASK: Optional[asyncio.Task] = None
SEQ = 0
SEQ_LOCK = asyncio.Lock()

TEXT_LIMITER = AsyncLimiter(max(int(TEXT_RATE), 1), 1.0)
MEDIA_LIMITER = AsyncLimiter(max(int(MEDIA_RATE), 1), 1.0)

PENDING_JOBS: Dict[str, int] = {}
PENDING_LOCK = asyncio.Lock()


async def enqueue(job: dict, source_key: str, priority: int = 10):
    global SEQ
    async with SEQ_LOCK:
        SEQ += 1
        seq = SEQ

    job["source_key"] = source_key

    async with PENDING_LOCK:
        PENDING_JOBS[source_key] = PENDING_JOBS.get(source_key, 0) + 1

    await SEND_QUEUE.put((priority, seq, job))


async def _pending_done(source_key: str):
    async with PENDING_LOCK:
        cur = PENDING_JOBS.get(source_key, 0)
        if cur <= 1:
            PENDING_JOBS.pop(source_key, None)
        else:
            PENDING_JOBS[source_key] = cur - 1


async def sender_loop(app):
    bot = app.bot
    while True:
        priority, seq, job = await SEND_QUEUE.get()
        source_key = job.get("source_key", "unknown")

        try:
            async with TEXT_LIMITER:
                await tg_retry(lambda: send_daily_header_if_needed(bot))

            kind = job.get("kind")

            if kind == "album":
                photos: List[str] = job.get("photos", [])
                cap_html: Optional[str] = job.get("caption_html")
                cap_plain: Optional[str] = job.get("caption_plain")

                async with MEDIA_LIMITER:
                    try:
                        media = []
                        for i, fid in enumerate(photos):
                            if i == 0 and cap_html:
                                media.append(InputMediaPhoto(media=fid, caption=cap_html, parse_mode=ParseMode.HTML))
                            else:
                                media.append(InputMediaPhoto(media=fid))
                        await tg_retry(lambda: bot.send_media_group(chat_id=OUTPUT_CHAT_ID_INT, media=media))
                    except BadRequest:
                        media2 = []
                        for i, fid in enumerate(photos):
                            if i == 0 and cap_plain:
                                media2.append(InputMediaPhoto(media=fid, caption=cap_plain))
                            else:
                                media2.append(InputMediaPhoto(media=fid))
                        await tg_retry(lambda: bot.send_media_group(chat_id=OUTPUT_CHAT_ID_INT, media=media2))

            elif kind == "photo":
                fid = job["photo"]
                cap_html = job.get("caption_html")
                cap_plain = job.get("caption_plain")

                async with MEDIA_LIMITER:
                    try:
                        await tg_retry(lambda: bot.send_photo(
                            chat_id=OUTPUT_CHAT_ID_INT,
                            photo=fid,
                            caption=(cap_html if cap_html else None),
                            parse_mode=(ParseMode.HTML if cap_html else None),
                        ))
                    except BadRequest:
                        await tg_retry(lambda: bot.send_photo(
                            chat_id=OUTPUT_CHAT_ID_INT,
                            photo=fid,
                            caption=(cap_plain if cap_plain else None),
                        ))

            elif kind == "text":
                html_txt = job["text_html"]
                plain_txt = job["text_plain"]

                async with TEXT_LIMITER:
                    try:
                        await tg_retry(lambda: bot.send_message(
                            chat_id=OUTPUT_CHAT_ID_INT,
                            text=html_txt,
                            parse_mode=ParseMode.HTML,
                            disable_web_page_preview=True,
                        ))
                    except BadRequest:
                        await tg_retry(lambda: bot.send_message(
                            chat_id=OUTPUT_CHAT_ID_INT,
                            text=plain_txt,
                            disable_web_page_preview=True,
                        ))

        except Forbidden:
            log.error("Forbidden: il bot non ha permessi nel gruppo. Rendilo admin.")
        except Exception as e:
            log.exception("Sender error: %s", e)
        finally:
            await _pending_done(source_key)
            SEND_QUEUE.task_done()
            asyncio.create_task(try_finalize_batch_fast(source_key))


async def post_init(app):
    global SENDER_TASK
    if SENDER_TASK is None or SENDER_TASK.done():
        SENDER_TASK = asyncio.create_task(sender_loop(app), name="sender-loop")
        log.info("Sender loop started.")

# =========================
# SOURCE / FORWARD HELPERS
# =========================
def extract_forward_source(msg) -> tuple[str, str]:
    fo = getattr(msg, "forward_origin", None)
    if fo:
        chat = getattr(fo, "chat", None)
        if chat:
            title = getattr(chat, "title", None) or (f"@{chat.username}" if getattr(chat, "username", None) else None)
            if not title:
                title = f"Chat {chat.id}"
            return f"chat:{chat.id}", title
        user = getattr(fo, "sender_user", None)
        if user:
            label = f"@{user.username}" if getattr(user, "username", None) else (user.full_name or f"User {user.id}")
            return f"user:{user.id}", label

    fchat = getattr(msg, "forward_from_chat", None)
    if fchat:
        label = getattr(fchat, "title", None) or (f"@{fchat.username}" if getattr(fchat, "username", None) else f"Chat {fchat.id}")
        return f"chat:{fchat.id}", label

    fuser = getattr(msg, "forward_from", None)
    if fuser:
        label = f"@{fuser.username}" if getattr(fuser, "username", None) else (fuser.full_name or f"User {user.id}")
        return f"user:{fuser.id}", label

    return "hidden", "Origine nascosta (forward anonimo)"


def is_forward(msg) -> bool:
    return bool(
        getattr(msg, "forward_origin", None) or getattr(msg, "forward_date", None) or
        getattr(msg, "forward_from", None) or getattr(msg, "forward_from_chat", None)
    )


def msg_source(msg) -> Tuple[str, str]:
    if is_forward(msg):
        return extract_forward_source(msg)
    return "manual", "Manuale (non-forward)"

# =========================
# ALBUM BUFFER (media_group) - debounce + watchdog
# =========================
@dataclass
class MediaGroupBuffer:
    media_group_id: str
    source_key: str
    source_label: str
    photo_file_ids: List[str] = field(default_factory=list)
    caption_html: Optional[str] = None
    first_ts: float = 0.0
    last_ts: float = 0.0
    task: Optional[asyncio.Task] = None


BUFFERS: Dict[str, MediaGroupBuffer] = {}
LOCK = asyncio.Lock()


async def finalize_media_group(key: str):
    try:
        while True:
            await asyncio.sleep(MEDIA_GROUP_WAIT)
            async with LOCK:
                buf = BUFFERS.get(key)
                if not buf:
                    return

                now = asyncio.get_event_loop().time()
                quiet = (now - buf.last_ts) >= MEDIA_GROUP_WAIT
                too_old = (now - buf.first_ts) >= MAX_ALBUM_AGE

                if quiet or too_old:
                    BUFFERS.pop(key, None)
                    break

        unique_photos = list(dict.fromkeys(buf.photo_file_ids))
        if not unique_photos:
            return

        # ✅ max 4 random
        photos = random.sample(unique_photos, MAX_PHOTOS) if len(unique_photos) > MAX_PHOTOS else unique_photos

        cap_html = None
        cap_plain = None
        cap_src = buf.caption_html or ""
        if cap_src:
            cap_html2, _ = rewrite_html_message_safe(cap_src)
            cap_plain2 = html_to_plain_with_links(cap_html2, max_len=TEXT_LIMIT)

            cap_plain2 = truncate_text(cap_plain2, ALBUM_CAPTION_LIMIT)
            if len(cap_html2) <= ALBUM_CAPTION_LIMIT:
                cap_html = cap_html2
                cap_plain = cap_plain2
            else:
                cap_html = None
                cap_plain = cap_plain2

        await enqueue({
            "kind": "album",
            "photos": photos,
            "caption_html": cap_html,
            "caption_plain": cap_plain,
        }, source_key=buf.source_key, priority=10)

    except Exception as e:
        log.exception("finalize_media_group failed: %s", e)

# =========================
# BATCH TRACKER
# =========================
def emoji_for_source(source_key: str) -> str:
    palette = ["🟣","🟢","🟠","🔵","🔴","🟡","🟤","⚫️"]
    return palette[abs(hash(source_key)) % len(palette)]


def make_batch_id() -> str:
    return "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(4))


@dataclass
class BatchBuffer:
    source_key: str
    source_label: str
    batch_id: str = field(default_factory=make_batch_id)
    emoji: str = "🟣"

    total_msgs: int = 0
    text_msgs: int = 0
    photo_msgs: int = 0
    albums: int = 0
    posts: int = 0

    seen_albums: set = field(default_factory=set)
    pending_text_ts: Optional[float] = None

    last_ts: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    task: Optional[asyncio.Task] = None
    finalized: bool = False


BATCHES: Dict[str, BatchBuffer] = {}
BATCH_LOCK = asyncio.Lock()


async def finalize_batch(key: str):
    await asyncio.sleep(BATCH_WAIT)
    await finalize_batch_now(key, reason="timer")


async def finalize_batch_now(key: str, reason: str = "fast"):
    async with BATCH_LOCK:
        b = BATCHES.get(key)
        if not b or b.finalized:
            return

        now = asyncio.get_event_loop().time()
        if reason != "timer" and (now - b.last_ts) < FAST_FINALIZE_IDLE:
            return

        b.finalized = True
        BATCHES.pop(key, None)

    if b.pending_text_ts is not None:
        b.posts += 1
        b.pending_text_ts = None

    STATE["last_batch"] = {
        "when": datetime.now(ZONE).isoformat(),
        "source": b.source_label,
        "batch_id": b.batch_id,
        "posts": b.posts,
        "total": b.total_msgs,
        "texts": b.text_msgs,
        "photos": b.photo_msgs,
        "albums": b.albums,
        "emoji": b.emoji,
        "reason": reason,
    }
    save_state(STATE)

    if b.total_msgs < BATCH_MIN_MESSAGES:
        return

    txt = (
        f"{b.emoji} <b>INOLTRO COMPLETATO — #{b.batch_id}</b>\n"
        f"📩 <b>Da:</b> {html.escape(b.source_label)}\n"
        f"✅ <b>Post:</b> {b.posts}\n"
        f"🧾 <b>Messaggi totali:</b> {b.total_msgs} | 💬 <b>Testi:</b> {b.text_msgs} | 🖼️ <b>Foto:</b> {b.photo_msgs} | 🧩 <b>Album:</b> {b.albums}\n"
        f"⏱️ <b>Ora:</b> {datetime.now(ZONE).strftime('%H:%M:%S')}"
    )

    plain = re.sub(r"<[^>]+>", "", html.unescape(txt))
    plain = truncate_text(plain, TEXT_LIMIT)
    txt = truncate_text(txt, TEXT_LIMIT)

    await enqueue(
        {"kind": "text", "text_html": txt, "text_plain": plain},
        source_key=key,
        priority=0
    )


async def try_finalize_batch_fast(source_key: str):
    # anti-race
    await asyncio.sleep(0.2)

    # pending jobs deve essere 0
    async with PENDING_LOCK:
        pending = PENDING_JOBS.get(source_key, 0)
    if pending != 0:
        return

    # nessun album buffer aperto
    async with LOCK:
        for buf in BUFFERS.values():
            if buf.source_key == source_key:
                return

    # aspetta idle reale
    while True:
        async with BATCH_LOCK:
            b = BATCHES.get(source_key)
            if not b or b.finalized:
                return
            now = asyncio.get_event_loop().time()
            idle = now - b.last_ts

        if idle >= FAST_FINALIZE_IDLE:
            break

        await asyncio.sleep((FAST_FINALIZE_IDLE - idle) + 0.15)

        async with PENDING_LOCK:
            pending = PENDING_JOBS.get(source_key, 0)
        if pending != 0:
            return

        async with LOCK:
            for buf in BUFFERS.values():
                if buf.source_key == source_key:
                    return

    await finalize_batch_now(source_key, reason="fast")


async def batch_touch(msg):
    if not is_forward(msg):
        return

    source_key, source_label = extract_forward_source(msg)
    key = source_key
    now = asyncio.get_event_loop().time()

    async with BATCH_LOCK:
        b = BATCHES.get(key)
        if not b:
            b = BatchBuffer(source_key=source_key, source_label=source_label)
            b.emoji = emoji_for_source(source_key)
            BATCHES[key] = b

        b.total_msgs += 1
        b.last_ts = now

        if msg.media_group_id:
            if msg.photo:
                b.photo_msgs += 1

            mgid = str(msg.media_group_id)
            if mgid not in b.seen_albums:
                b.seen_albums.add(mgid)
                b.albums += 1

                if b.pending_text_ts and (now - b.pending_text_ts) <= PAIR_WINDOW:
                    b.posts += 1
                    b.pending_text_ts = None
                else:
                    b.posts += 1

        elif msg.photo:
            b.photo_msgs += 1
            if b.pending_text_ts and (now - b.pending_text_ts) <= PAIR_WINDOW:
                b.posts += 1
                b.pending_text_ts = None
            else:
                b.posts += 1

        elif msg.text or msg.caption:
            b.text_msgs += 1
            b.pending_text_ts = now

        if b.task and not b.task.done():
            b.task.cancel()
        b.task = asyncio.create_task(finalize_batch(key))

# =========================
# COMMANDS
# =========================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Bot attivo (conversione ref cnfans/mulebuy + queue + batch log).",
        disable_web_page_preview=True,
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_USER_ID_INT:
        return

    async with PENDING_LOCK:
        pending_total = sum(PENDING_JOBS.values())

    last = STATE.get("last_batch")
    last_line = ""
    if last:
        last_line = (
            f"\n\nUltimo batch: {last.get('emoji','')} #{last.get('batch_id')} | "
            f"Post {last.get('posts')} | Msg {last.get('total')} | Da: {last.get('source')} | ({last.get('reason','?')})"
        )

    await update.message.reply_text(
        f"📦 Coda: {SEND_QUEUE.qsize()} / {QUEUE_MAX} | Pending: {pending_total}\n"
        f"⚙️ MAX_PHOTOS: {MAX_PHOTOS} | MEDIA_GROUP_WAIT: {MEDIA_GROUP_WAIT}s | MAX_ALBUM_AGE: {MAX_ALBUM_AGE}s\n"
        f"🚀 TEXT_RATE: {TEXT_RATE}/s | MEDIA_RATE: {MEDIA_RATE}/s | IDLE: {FAST_FINALIZE_IDLE}s\n"
        f"🧾 BATCH_WAIT(fallback): {BATCH_WAIT}s | PAIR_WINDOW: {PAIR_WINDOW}s"
        f"{last_line}",
        disable_web_page_preview=True,
    )


async def cmd_lastbatch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_USER_ID_INT:
        return
    last = STATE.get("last_batch")
    if not last:
        await update.message.reply_text("Nessun batch ancora registrato.")
        return
    txt = (
        f"{last.get('emoji','')} <b>ULTIMO BATCH — #{last.get('batch_id')}</b>\n"
        f"📩 <b>Da:</b> {html.escape(last.get('source','?'))}\n"
        f"✅ <b>Post:</b> {last.get('posts','?')}\n"
        f"🧾 <b>Messaggi totali:</b> {last.get('total','?')} | 💬 <b>Testi:</b> {last.get('texts','?')} | "
        f"🖼️ <b>Foto:</b> {last.get('photos','?')} | 🧩 <b>Album:</b> {last.get('albums','?')}\n"
        f"⏱️ <b>Quando:</b> {html.escape(last.get('when',''))}\n"
        f"🧠 <b>Trigger:</b> {html.escape(last.get('reason','?'))}"
    )
    await update.message.reply_text(txt, parse_mode=ParseMode.HTML, disable_web_page_preview=True)


async def cmd_flush(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_USER_ID_INT:
        return
    keys = list(BATCHES.keys())
    for k in keys:
        await finalize_batch_now(k, reason="manual-flush")
    await update.message.reply_text("✅ Batch chiusi manualmente (/flush).")


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

    if update.effective_chat.type != "private":
        return

    if update.effective_user.id != OWNER_USER_ID_INT:
        return

    await batch_touch(msg)

    source_key, source_label = msg_source(msg)

    # Album buffer
    if msg.media_group_id:
        key = str(msg.media_group_id)
        now = asyncio.get_event_loop().time()
        async with LOCK:
            buf = BUFFERS.get(key)
            if not buf:
                buf = MediaGroupBuffer(
                    media_group_id=key,
                    source_key=source_key,
                    source_label=source_label,
                    first_ts=now,
                    last_ts=now,
                )
                BUFFERS[key] = buf

            buf.last_ts = now

            if msg.photo:
                buf.photo_file_ids.append(msg.photo[-1].file_id)

            if not buf.caption_html and msg.caption_html:
                buf.caption_html = msg.caption_html

            if not buf.task or buf.task.done():
                buf.task = asyncio.create_task(finalize_media_group(key))
        return

    # ignora video
    if msg.video:
        return

    # Foto singola
    if msg.photo:
        cap_html = None
        cap_plain = None
        if msg.caption_html:
            cap_html2, _ = rewrite_html_message_safe(msg.caption_html)
            cap_plain2 = html_to_plain_with_links(cap_html2, max_len=TEXT_LIMIT)

            cap_plain2 = truncate_text(cap_plain2, PHOTO_CAPTION_LIMIT)
            if len(cap_html2) <= PHOTO_CAPTION_LIMIT:
                cap_html = cap_html2
                cap_plain = cap_plain2
            else:
                cap_html = None
                cap_plain = cap_plain2

        await enqueue({
            "kind": "photo",
            "photo": msg.photo[-1].file_id,
            "caption_html": cap_html,
            "caption_plain": cap_plain,
        }, source_key=source_key, priority=10)
        return

    # Testo: pubblica SOLO se c’è almeno una modifica (ref forzato)
    if msg.text_html:
        rewritten_html, changed = rewrite_html_message_safe(msg.text_html)
        if not changed:
            return

        plain = html_to_plain_with_links(rewritten_html, max_len=TEXT_LIMIT)

        if len(rewritten_html) > TEXT_LIMIT:
            rewritten_html = None
            plain = truncate_text(plain, TEXT_LIMIT)
        else:
            rewritten_html = truncate_text(rewritten_html, TEXT_LIMIT)
            plain = truncate_text(plain, TEXT_LIMIT)

        await enqueue({
            "kind": "text",
            "text_html": (rewritten_html if rewritten_html else plain),
            "text_plain": plain,
        }, source_key=source_key, priority=10)
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

    concurrent_updates = int(os.getenv("CONCURRENT_UPDATES", "12"))

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .request(request)
        .concurrent_updates(concurrent_updates)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("lastbatch", cmd_lastbatch))
    app.add_handler(CommandHandler("flush", cmd_flush))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle))

    log.info("Bot running (force-ref only: mulebuy + cnfans)...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )


if __name__ == "__main__":
    main()


