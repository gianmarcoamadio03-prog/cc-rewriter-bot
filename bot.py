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
from telegram.request import HTTPXRequest
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


def env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    return float(raw)


BOT_TOKEN = env_str("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Manca BOT_TOKEN nel .env")

OWNER_USER_ID_INT = env_int("OWNER_USER_ID", None)

# Gruppo coda interna: POST PRONTI
POST_PRONTI_CHAT_ID_INT = env_int("POST_PRONTI_CHAT_ID", None)
if POST_PRONTI_CHAT_ID_INT is None:
    POST_PRONTI_CHAT_ID_INT = env_int("OUTPUT_CHAT_ID", None)

# Destinazione finale: canale/gruppo pubblico
FINAL_CHAT_ID_INT = env_int("BEST_FIND_CHAT_ID", None)
if FINAL_CHAT_ID_INT is None:
    FINAL_CHAT_ID_INT = env_int("FINAL_CHAT_ID", None)

MULEBUY_REF = env_str("MULEBUY_REF", "200836051")
CNFANS_REF = env_str("CNFANS_REF", "222394")

TZ = env_str("TZ", "Europe/Rome")
MAX_PHOTOS = env_int("MAX_PHOTOS", 4) or 4
MEDIA_GROUP_WAIT = env_float("MEDIA_GROUP_WAIT", 1.2)
HTTP_TIMEOUT = env_float("HTTP_TIMEOUT", 60.0)

TEXT_LIMIT = env_int("TEXT_LIMIT", 3900) or 3900
PHOTO_CAPTION_LIMIT = env_int("PHOTO_CAPTION_LIMIT", 950) or 950
STATE_FILE = env_str("STATE_FILE", "state.json")

SEND_DELAY_MIN = env_float("SEND_DELAY_MIN", 0.8)
SEND_DELAY_MAX = env_float("SEND_DELAY_MAX", 1.8)

ZONE = ZoneInfo(TZ)

STATE_LOCK = asyncio.Lock()
BUFFERS_LOCK = asyncio.Lock()

URL_RE = re.compile(r"(https?://[^\s<>\]]+)", re.IGNORECASE)
HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)
TRAILING_PUNCT = ".,;:!?)\u201d\u2019]"
ID_CMD_RE = re.compile(r"^/id(?:@[A-Za-z0-9_]+)?$", re.IGNORECASE)
CRAV_TEXT_RE = re.compile(
    r"(?i)(?:https?://)?(?:www\.)?cravattacinese\.com"
)


# =========================
# STATE
# =========================
def base_state() -> dict:
    return {
        "next_post_id": 1,
        "queue": [],
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

    state = base_state()
    state.update(data or {})
    state["stats"] = {**base_state()["stats"], **(state.get("stats") or {})}
    state["queue"] = state.get("queue") or []

    if not isinstance(state.get("next_post_id"), int):
        state["next_post_id"] = 1

    return state


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


STATE = load_state()


def now_iso() -> str:
    return datetime.now(ZONE).isoformat(timespec="seconds")


async def queue_add(item: dict) -> int:
    async with STATE_LOCK:
        row = dict(item)
        row["id"] = STATE["next_post_id"]
        STATE["next_post_id"] += 1
        STATE["queue"].append(row)
        STATE["stats"]["converted_total"] += 1
        save_state(STATE)
        return row["id"]


async def queue_list_pending() -> List[dict]:
    async with STATE_LOCK:
        return list(STATE["queue"])


async def queue_count() -> int:
    async with STATE_LOCK:
        return len(STATE["queue"])


async def queue_mark_sent_and_remove(item_id: int) -> None:
    async with STATE_LOCK:
        new_queue = [x for x in STATE["queue"] if x.get("id") != item_id]
        if len(new_queue) != len(STATE["queue"]):
            STATE["queue"] = new_queue
            STATE["stats"]["sent_total"] += 1
            save_state(STATE)


# =========================
# HELPERS
# =========================
def owner_only(update: Update) -> bool:
    user = update.effective_user
    return bool(
        OWNER_USER_ID_INT is not None
        and user is not None
        and user.id == OWNER_USER_ID_INT
    )


async def require_owner(update: Update) -> bool:
    msg = update.effective_message
    if not msg:
        return False

    if OWNER_USER_ID_INT is None:
        await msg.reply_text("⚠️ OWNER_USER_ID non impostato nel .env.")
        return False

    if not owner_only(update):
        return False

    return True


def commands_chat_ok(update: Update) -> bool:
    chat = update.effective_chat
    if not chat:
        return False
    if chat.type == "private":
        return True
    return POST_PRONTI_CHAT_ID_INT is not None and chat.id == POST_PRONTI_CHAT_ID_INT


async def require_owner_and_command_chat(update: Update) -> bool:
    msg = update.effective_message
    if not msg:
        return False

    if not await require_owner(update):
        return False

    if not commands_chat_ok(update):
        await msg.reply_text(
            "⚠️ Usa questo comando in privato oppure nel gruppo POST PRONTI."
        )
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


def normalize_visible_plain_text(text: str) -> tuple[str, bool]:
    if not text:
        return text, False

    changed = False

    def repl(match):
        nonlocal changed
        changed = True
        return "www.cravattacinese.com"

    out = CRAV_TEXT_RE.sub(repl, text)
    return out, changed


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

    def repl_visible_url(match):
        nonlocal changed
        old_with_trail = match.group(1)
        old, trail = strip_trailing_punct(old_with_trail)
        unescaped = html.unescape(old)
        new = transform_url(unescaped)
        if new != unescaped:
            changed = True
        return html.escape(new, quote=False) + trail

    for i in range(0, len(parts), 2):
        parts[i] = URL_RE.sub(repl_visible_url, parts[i])
        normalized, text_changed = normalize_visible_plain_text(parts[i])
        if text_changed:
            changed = True
            parts[i] = normalized

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


# =========================
# HOLDING GROUP / POST PRONTI
# =========================
async def publish_to_post_pronti(bot, item: dict) -> List[int]:
    if POST_PRONTI_CHAT_ID_INT is None:
        raise RuntimeError("POST_PRONTI_CHAT_ID non configurato")

    kind = item.get("kind")

    if kind == "text":
        msg = await with_retry(lambda: bot.send_message(
            chat_id=POST_PRONTI_CHAT_ID_INT,
            text=truncate_text(item.get("html", ""), TEXT_LIMIT),
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        ))
        return [msg.message_id]

    if kind == "album":
        photos = list(dict.fromkeys(item.get("photos") or []))
        caption_html = truncate_text(item.get("caption_html", ""), PHOTO_CAPTION_LIMIT)

        if len(photos) == 1:
            msg = await with_retry(lambda: bot.send_photo(
                chat_id=POST_PRONTI_CHAT_ID_INT,
                photo=photos[0],
                caption=caption_html or None,
                parse_mode=ParseMode.HTML if caption_html else None,
            ))
            return [msg.message_id]

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

        msgs = await with_retry(lambda: bot.send_media_group(
            chat_id=POST_PRONTI_CHAT_ID_INT,
            media=media,
        ))
        return [m.message_id for m in msgs]

    return []


async def delete_from_post_pronti(bot, message_ids: List[int]) -> None:
    if POST_PRONTI_CHAT_ID_INT is None:
        return

    for mid in message_ids or []:
        try:
            await with_retry(lambda mid=mid: bot.delete_message(
                chat_id=POST_PRONTI_CHAT_ID_INT,
                message_id=mid,
            ))
        except Exception:
            log.exception("Impossibile cancellare message_id=%s da POST PRONTI", mid)


async def queue_and_publish_to_post_pronti(bot, item: dict) -> bool:
    try:
        holding_ids = await publish_to_post_pronti(bot, item)
    except Exception:
        log.exception("Errore pubblicazione in POST PRONTI")
        return False

    if not holding_ids:
        return False

    item = dict(item)
    item["holding_message_ids"] = holding_ids
    item["created_at"] = now_iso()
    await queue_add(item)
    return True


# =========================
# FINAL SEND
# =========================
async def send_to_final_destination(bot, item: dict) -> bool:
    if FINAL_CHAT_ID_INT is None:
        return False

    kind = item.get("kind")

    try:
        if kind == "text":
            await with_retry(lambda: bot.send_message(
                chat_id=FINAL_CHAT_ID_INT,
                text=truncate_text(item.get("html", ""), TEXT_LIMIT),
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
                    chat_id=FINAL_CHAT_ID_INT,
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
                chat_id=FINAL_CHAT_ID_INT,
                media=media,
            ))
            return True

        return False

    except Forbidden:
        log.exception("Forbidden while sending final item id=%s", item.get("id"))
        return False
    except BadRequest:
        log.exception("BadRequest while sending final item id=%s", item.get("id"))
        return False
    except Exception:
        log.exception("Unexpected error while sending final item id=%s", item.get("id"))
        return False


# =========================
# UNIVERSAL /id
# =========================
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


# =========================
# COMMANDS
# =========================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return

    await msg.reply_text(
        "✅ Bot attivo.\n"
        "Privato col bot: inoltra i post da convertire.\n"
        "Il bot li manda in POST PRONTI.\n"
        "Comandi:\n"
        "/totale\n"
        "/invio 50\n"
        "/invia 50\n"
        "/id"
    )


async def cmd_totale(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return

    if not await require_owner_and_command_chat(update):
        return

    pending = await queue_count()
    await msg.reply_text(
        f"📦 Post attualmente in coda in POST PRONTI: <b>{pending}</b>",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


async def cmd_invio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if not msg:
        return

    if not await require_owner_and_command_chat(update):
        return

    if FINAL_CHAT_ID_INT is None:
        await msg.reply_text(
            "⚠️ BEST_FIND_CHAT_ID / FINAL_CHAT_ID non impostato nel .env."
        )
        return

    pending = await queue_list_pending()
    if not pending:
        await msg.reply_text("📭 Nessun post in coda.")
        return

    limit = len(pending)
    if context.args:
        try:
            limit = int(context.args[0])
            if limit <= 0:
                raise ValueError
        except ValueError:
            await msg.reply_text("Uso corretto: /invio 50")
            return

    limit = min(limit, len(pending))
    selected = random.sample(pending, k=limit)

    sent_ok = 0
    failed = 0

    for idx, item in enumerate(selected, start=1):
        ok = await send_to_final_destination(context.bot, item)
        if ok:
            await delete_from_post_pronti(context.bot, item.get("holding_message_ids") or [])
            await queue_mark_sent_and_remove(item["id"])
            sent_ok += 1
        else:
            failed += 1

        if idx < len(selected):
            await asyncio.sleep(random.uniform(SEND_DELAY_MIN, SEND_DELAY_MAX))

    remaining = await queue_count()

    await msg.reply_text(
        f"✅ Inviati: <b>{sent_ok}</b>\n"
        f"❌ Falliti: <b>{failed}</b>\n"
        f"📦 Rimasti in coda: <b>{remaining}</b>\n"
        f"🎲 Ordine: <b>casuale / sfalsato</b>",
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


# =========================
# PRIVATE INPUT -> CONVERT -> POST PRONTI
# =========================
async def finalize_media_group(key: str, bot) -> None:
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

        item = {
            "kind": "album",
            "photos": photos,
            "caption_html": truncate_text(rewritten_html, PHOTO_CAPTION_LIMIT),
        }
        await queue_and_publish_to_post_pronti(bot, item)
        return


async def handle_private_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    msg = update.message

    if not chat or not msg:
        return

    if chat.type != "private":
        return

    if not await require_owner(update):
        return

    if POST_PRONTI_CHAT_ID_INT is None:
        await msg.reply_text("⚠️ POST_PRONTI_CHAT_ID non impostato nel .env.")
        return

    # album
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
                buf.task = context.application.create_task(
                    finalize_media_group(key, context.bot)
                )
        return

    # foto singola
    if msg.photo:
        caption_html = msg.caption_html or ""
        rewritten_html, changed = rewrite_html_message_safe(caption_html)
        if not changed:
            return

        item = {
            "kind": "album",
            "photos": [msg.photo[-1].file_id],
            "caption_html": truncate_text(rewritten_html, PHOTO_CAPTION_LIMIT),
        }
        await queue_and_publish_to_post_pronti(context.bot, item)
        return

    # testo
    text_html = msg.text_html or ""
    if text_html:
        rewritten_html, changed = rewrite_html_message_safe(text_html)
        if not changed:
            return

        item = {
            "kind": "text",
            "html": truncate_text(rewritten_html, TEXT_LIMIT),
        }
        await queue_and_publish_to_post_pronti(context.bot, item)
        return


# =========================
# ERROR
# =========================
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.exception("HANDLER ERROR: %s", context.error)


# =========================
# MAIN
# =========================
def main():
    request = HTTPXRequest(
        connection_pool_size=20,
        read_timeout=HTTP_TIMEOUT,
        write_timeout=HTTP_TIMEOUT,
        connect_timeout=HTTP_TIMEOUT,
        pool_timeout=HTTP_TIMEOUT,
    )

    app = ApplicationBuilder().token(BOT_TOKEN).request(request).build()

    # /id in ogni tipo di chat
    app.add_handler(TypeHandler(Update, catch_id_everywhere), group=-1)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler(["totale", "quanti"], cmd_totale))
    app.add_handler(CommandHandler(["invio", "invia"], cmd_invio))

    # input privato del bot
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_private_input))

    app.add_error_handler(on_error)

    log.info("Bot running: privato -> POST PRONTI -> /invio N -> canale finale")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
