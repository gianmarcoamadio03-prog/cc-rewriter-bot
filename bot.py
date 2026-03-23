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
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from telegram import Update, InputMediaPhoto
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut, NetworkError, BadRequest, Forbidden
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    TypeHandler,
    filters,
)

# =========================
# ENV + LOG
# =========================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("cc-rewriter-bot")

BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
OWNER_USER_ID = (os.getenv("OWNER_USER_ID") or "").strip()
OUTPUT_CHAT_ID = (os.getenv("OUTPUT_CHAT_ID") or "").strip()
BEST_FIND_CHAT_ID = (os.getenv("BEST_FIND_CHAT_ID") or "").strip()

MULEBUY_REF = (os.getenv("MULEBUY_REF") or "200836051").strip()
CNFANS_REF = (os.getenv("CNFANS_REF") or "222394").strip()

TZ = (os.getenv("TZ") or "Europe/Rome").strip()
MAX_PHOTOS = int((os.getenv("MAX_PHOTOS") or "4").strip())
MEDIA_GROUP_WAIT = float((os.getenv("MEDIA_GROUP_WAIT") or "1.2").strip())
STATE_FILE = (os.getenv("STATE_FILE") or "state.json").strip()

TEXT_LIMIT = int((os.getenv("TEXT_LIMIT") or "3900").strip())
PHOTO_CAPTION_LIMIT = int((os.getenv("PHOTO_CAPTION_LIMIT") or "950").strip())

if not BOT_TOKEN:
    raise RuntimeError("Manca BOT_TOKEN nel .env")

OWNER_USER_ID_INT = int(OWNER_USER_ID) if OWNER_USER_ID else None
BEST_FIND_CHAT_ID_INT = (
    int(BEST_FIND_CHAT_ID) if BEST_FIND_CHAT_ID
    else (int(OUTPUT_CHAT_ID) if OUTPUT_CHAT_ID else None)
)

ZONE = ZoneInfo(TZ)

STATE_LOCK = asyncio.Lock()
BUFFERS_LOCK = asyncio.Lock()

URL_RE = re.compile(r"(https?://[^\s<>\]]+)", re.IGNORECASE)
HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)
TRAILING_PUNCT = ".,;:!?)\u201d\u2019]"
ID_CMD_RE = re.compile(r"^/id(?:@[A-Za-z0-9_]+)?$", re.IGNORECASE)


# =========================
# STATE
# =========================
def base_state() -> dict:
    return {
        "next_post_id": 1,
        "pending_posts": [],
        "stats": {
            "converted_total": 0,
            "sent_total": 0,
        },
    }


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return base_state()

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return base_state()

    out = base_state()
    out.update(data or {})
    out["stats"] = {**base_state()["stats"], **(out.get("stats") or {})}
    out["pending_posts"] = out.get("pending_posts") or []

    if not isinstance(out.get("next_post_id"), int):
        out["next_post_id"] = 1

    return out


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


STATE = load_state()


def now_iso() -> str:
    return datetime.now(ZONE).isoformat(timespec="seconds")


async def add_pending_post(item: dict) -> int:
    async with STATE_LOCK:
        row = dict(item)
        row["id"] = STATE["next_post_id"]
        STATE["next_post_id"] += 1
        STATE["pending_posts"].append(row)
        STATE["stats"]["converted_total"] += 1
        save_state(STATE)
        return len(STATE["pending_posts"])


async def get_stats() -> dict:
    async with STATE_LOCK:
        return {
            "pending": len(STATE["pending_posts"]),
            "converted_total": STATE["stats"]["converted_total"],
            "sent_total": STATE["stats"]["sent_total"],
        }


# =========================
# HELPERS
# =========================
def owner_only(update: Update) -> bool:
    user = update.effective_user
    return bool(
        OWNER_USER_ID_INT is not None and
        user is not None and
        user.id == OWNER_USER_ID_INT
    )


async def require_owner(update: Update) -> bool:
    msg = update.effective_message
    if not msg:
        return False

    if OWNER_USER_ID_INT is None:
        await msg.reply_text(
            "⚠️ OWNER_USER_ID non impostato.\n"
            "Usa /id in privato col bot, copia il tuo User ID nel file .env come OWNER_USER_ID e riavvia il bot."
        )
        return False

    if not owner_only(update):
        return False

    return True


def truncate_text(text: str, limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def with_retry(coro_factory, retries: int = 5):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return await coro_factory()
        except RetryAfter as e:
            wait_s = int(getattr(e, "retry_after", 1)) + 1
            await asyncio.sleep(wait_s)
            last_exc = e
        except (TimedOut, NetworkError) as e:
            wait_s = min(2 ** attempt, 8)
            await asyncio.sleep(wait_s)
            last_exc = e
        except Exception as e:
            last_exc = e
            break

    if last_exc:
        raise last_exc


# =========================
# URL REWRITE
# =========================
def strip_trailing_punct(url: str):
    trail = ""
    while url and url[-1] in TRAILING_PUNCT:
        trail = url[-1] + trail
        url = url[:-1]
    return url, trail


def normalize_netloc(netloc: str) -> str:
    n = (netloc or "").lower().strip()
    if n.startswith("www."):
        n = n[4:]
    if n.startswith("m."):
        n = n[2:]
    return n


def rebuild_url(parsed, netloc: Optional[str] = None, query: Optional[str] = None) -> str:
    return urlunparse((
        parsed.scheme or "https",
        netloc if netloc is not None else parsed.netloc,
        parsed.path,
        parsed.params,
        query if query is not None else parsed.query,
        parsed.fragment,
    ))


def transform_cravattacinese(url: str) -> str:
    parsed = urlparse(url)
    if normalize_netloc(parsed.netloc) != "cravattacinese.com":
        return url
    return rebuild_url(parsed, netloc="www.cravattacinese.com")


def transform_mulebuy(url: str) -> str:
    if not MULEBUY_REF:
        return url

    parsed = urlparse(url)
    if normalize_netloc(parsed.netloc) != "mulebuy.com":
        return url
    if not parsed.path.startswith("/product"):
        return url

    qlist = list(parse_qsl(parsed.query, keep_blank_values=True))
    qlist = [(k, v) for (k, v) in qlist if k.lower() != "ref"]
    qlist.append(("ref", MULEBUY_REF))
    return rebuild_url(parsed, query=urlencode(qlist, doseq=True))


def transform_cnfans(url: str) -> str:
    parsed = urlparse(url)
    if normalize_netloc(parsed.netloc) != "cnfans.com":
        return url
    if not parsed.path.startswith("/product"):
        return url

    qlist = list(parse_qsl(parsed.query, keep_blank_values=True))
    qlist = [(k, v) for (k, v) in qlist if k.lower() != "ref"]
    qlist.append(("ref", CNFANS_REF))
    return rebuild_url(parsed, query=urlencode(qlist, doseq=True))


def transform_url(url: str) -> str:
    out = transform_cravattacinese(url)
    out = transform_mulebuy(out)
    out = transform_cnfans(out)
    return out


def rewrite_html_message_safe(html_text: str) -> tuple[str, bool]:
    if not html_text:
        return "", False

    changed = False

    def href_repl(match):
        nonlocal changed
        old = html.unescape(match.group(1))
        new = transform_url(old)
        if new != old:
            changed = True
        return f'href="{html.escape(new, quote=True)}"'

    out = HREF_RE.sub(href_repl, html_text)
    parts = re.split(r"(<[^>]+>)", out)

    def repl_visible(match):
        nonlocal changed
        old_with_trail = match.group(1)
        old, trail = strip_trailing_punct(old_with_trail)
        unescaped = html.unescape(old)
        new = transform_url(unescaped)
        if new != unescaped:
            changed = True
        return html.escape(new, quote=False) + trail

    for i in range(0, len(parts), 2):
        parts[i] = URL_RE.sub(repl_visible, parts[i])

    return "".join(parts), changed


# =========================
# MEDIA GROUP BUFFER
# =========================
@dataclass
class MediaGroupBuffer:
    key: str
    photo_file_ids: List[str] = field(default_factory=list)
    caption_html: str = ""
    last_ts: float = 0.0
    task: Optional[asyncio.Task] = None


BUFFERS: Dict[str, MediaGroupBuffer] = {}


async def finalize_media_group(key: str):
    while True:
        await asyncio.sleep(MEDIA_GROUP_WAIT)

        async with BUFFERS_LOCK:
            buf = BUFFERS.get(key)
            if not buf:
                return

            idle = asyncio.get_running_loop().time() - buf.last_ts
            if idle < MEDIA_GROUP_WAIT:
                continue

            BUFFERS.pop(key, None)

        unique_photos = list(dict.fromkeys(buf.photo_file_ids))
        if not unique_photos:
            return

        photos = random.sample(unique_photos, MAX_PHOTOS) if len(unique_photos) > MAX_PHOTOS else unique_photos

        caption_html = buf.caption_html or ""
        rewritten_html, changed = rewrite_html_message_safe(caption_html)

        if not changed:
            return

        await add_pending_post({
            "kind": "album",
            "photos": photos,
            "caption_html": truncate_text(rewritten_html, PHOTO_CAPTION_LIMIT),
            "created_at": now_iso(),
        })
        return


# =========================
# SEND
# =========================
async def send_pending_item(bot, item: dict) -> bool:
    kind = item.get("kind")

    try:
        if kind == "text":
            text = truncate_text(item.get("html", ""), TEXT_LIMIT)
            if not text:
                return False

            await with_retry(lambda: bot.send_message(
                chat_id=BEST_FIND_CHAT_ID_INT,
                text=text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            ))
            return True

        if kind == "album":
            photos = list(dict.fromkeys(item.get("photos") or []))
            if not photos:
                return False

            caption_html = truncate_text(item.get("caption_html", ""), PHOTO_CAPTION_LIMIT)

            if len(photos) == 1:
                await with_retry(lambda: bot.send_photo(
                    chat_id=BEST_FIND_CHAT_ID_INT,
                    photo=photos[0],
                    caption=caption_html or None,
                    parse_mode=ParseMode.HTML if caption_html else None,
                ))
                return True

            media = []
            for i, file_id in enumerate(photos[:MAX_PHOTOS]):
                if i == 0 and caption_html:
                    media.append(InputMediaPhoto(
                        media=file_id,
                        caption=caption_html,
                        parse_mode=ParseMode.HTML,
                    ))
                else:
                    media.append(InputMediaPhoto(media=file_id))

            await with_retry(lambda: bot.send_media_group(
                chat_id=BEST_FIND_CHAT_ID_INT,
                media=media,
            ))
            return True

        return False

    except Forbidden:
        log.exception("Forbidden while sending item id=%s", item.get("id"))
        return False
    except BadRequest:
        log.exception("BadRequest while sending item id=%s", item.get("id"))
        return False
    except Exception:
        log.exception("Unexpected error while sending item id=%s", item.get("id"))
        return False


# =========================
# COMMANDS
# =========================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return

    if OWNER_USER_ID_INT is None:
        await msg.reply_text(
            "✅ Bot avviato.\n"
            "Prima configurazione:\n"
            "1) usa /id in privato col bot per leggere il tuo User ID\n"
            "2) mettilo nel file .env come OWNER_USER_ID\n"
            "3) usa /id nel gruppo o canale per leggere il Chat ID\n"
            "4) mettilo nel file .env come BEST_FIND_CHAT_ID oppure OUTPUT_CHAT_ID\n"
            "5) riavvia il bot"
        )
        return

    if not owner_only(update):
        return

    await msg.reply_text(
        "✅ Bot attivo.\n"
        "Inoltrami i post in privato.\n"
        "/totale = quanti ha convertito e quanti sono in coda\n"
        "/invio = invia la coda nel gruppo/canale in ordine casuale\n"
        "/id = mostra User ID e Chat ID della chat corrente",
        disable_web_page_preview=True,
    )


async def cmd_totale(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return

    if not await require_owner(update):
        return

    stats = await get_stats()
    await msg.reply_text(
        f"📦 In coda: <b>{stats['pending']}</b>\n"
        f"🔁 Convertiti totali: <b>{stats['converted_total']}</b>\n"
        f"🚀 Inviati totali: <b>{stats['sent_total']}</b>",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


async def cmd_invio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return

    if not await require_owner(update):
        return

    if BEST_FIND_CHAT_ID_INT is None:
        await msg.reply_text(
            "⚠️ BEST_FIND_CHAT_ID / OUTPUT_CHAT_ID non impostato.\n"
            "Usa /id nel gruppo o canale di destinazione, copia il Chat ID nel file .env e riavvia il bot."
        )
        return

    async with STATE_LOCK:
        pending = list(STATE["pending_posts"])

    if not pending:
        await msg.reply_text("📭 Nessun post in coda da inviare.")
        return

    random.shuffle(pending)

    sent_ids = set()
    sent_ok = 0

    for item in pending:
        ok = await send_pending_item(context.bot, item)
        if ok:
            sent_ids.add(item["id"])
            sent_ok += 1

    async with STATE_LOCK:
        if sent_ids:
            STATE["pending_posts"] = [
                row for row in STATE["pending_posts"]
                if row.get("id") not in sent_ids
            ]
            STATE["stats"]["sent_total"] += sent_ok
            save_state(STATE)

        remaining = len(STATE["pending_posts"])

    await msg.reply_text(
        f"✅ Inviati: <b>{sent_ok}</b>\n"
        f"📦 Rimasti in coda: <b>{remaining}</b>\n"
        f"🎲 Ordine: <b>casuale</b>",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    msg = update.effective_message
    user = update.effective_user

    if not chat or not msg:
        return

    sender_chat = getattr(msg, "sender_chat", None)

    lines = ["🆔 Dati correnti"]

    if user:
        lines.append(f"User ID: <code>{user.id}</code>")
    elif sender_chat:
        lines.append(f"Sender Chat ID: <code>{sender_chat.id}</code>")

    lines.append(f"Chat ID: <code>{chat.id}</code>")
    lines.append(f"Nome chat: <b>{html.escape(chat.title or chat.full_name or 'Chat privata')}</b>")
    lines.append(f"Tipo chat: <b>{html.escape(str(chat.type))}</b>")

    await context.bot.send_message(
        chat_id=chat.id,
        text="\n".join(lines),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


async def catch_id_everywhere(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg or not getattr(msg, "text", None):
        return

    text = msg.text.strip()
    if ID_CMD_RE.match(text):
        await cmd_id(update, context)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.exception("HANDLER ERROR: %s", context.error)


# =========================
# HANDLE POSTS
# =========================
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if OWNER_USER_ID_INT is None:
        return

    if not owner_only(update):
        return

    msg = update.message
    if not msg:
        return

    if msg.media_group_id and msg.photo:
        key = f"{msg.chat_id}:{msg.media_group_id}"

        async with BUFFERS_LOCK:
            buf = BUFFERS.get(key)
            if not buf:
                buf = MediaGroupBuffer(key=key)
                BUFFERS[key] = buf

            buf.photo_file_ids.append(msg.photo[-1].file_id)

            if msg.caption_html and not buf.caption_html:
                buf.caption_html = msg.caption_html

            buf.last_ts = asyncio.get_running_loop().time()

            if buf.task is None or buf.task.done():
                buf.task = context.application.create_task(finalize_media_group(key))
        return

    if msg.photo:
        caption_html = msg.caption_html or ""
        rewritten_html, changed = rewrite_html_message_safe(caption_html)

        if not changed:
            return

        await add_pending_post({
            "kind": "album",
            "photos": [msg.photo[-1].file_id],
            "caption_html": truncate_text(rewritten_html, PHOTO_CAPTION_LIMIT),
            "created_at": now_iso(),
        })
        return

    text_html = msg.text_html or ""
    if text_html:
        rewritten_html, changed = rewrite_html_message_safe(text_html)

        if not changed:
            return

        await add_pending_post({
            "kind": "text",
            "html": truncate_text(rewritten_html, TEXT_LIMIT),
            "created_at": now_iso(),
        })
        return


# =========================
# MAIN
# =========================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # intercetta /id in privato, gruppo, supergruppo e canale
    app.add_handler(TypeHandler(Update, catch_id_everywhere), group=-1)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("totale", cmd_totale))
    app.add_handler(CommandHandler("invio", cmd_invio))

    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle))
    app.add_error_handler(on_error)

    log.info("Bot running: queue manuale + /totale + /invio + /id + cravattacinese canonical")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
