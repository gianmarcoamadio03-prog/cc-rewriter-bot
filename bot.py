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
log = logging.getLogger("cc-rewriter-dm")

BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_USER_ID = os.getenv("OWNER_USER_ID")          # tuo user id telegram
OUTPUT_CHAT_ID = os.getenv("OUTPUT_CHAT_ID")        # chat id gruppo "POST PRONTI" (supergroup -100...)

MULEBUY_REF = os.getenv("MULEBUY_REF", "200836051")
CNFANS_REF = os.getenv("CNFANS_REF", "222394")

TZ = os.getenv("TZ", "Europe/Rome")
MAX_PHOTOS = int(os.getenv("MAX_PHOTOS", "4"))
MEDIA_GROUP_WAIT = float(os.getenv("MEDIA_GROUP_WAIT", "1.2"))

STATE_FILE = os.getenv("STATE_FILE", "state.json")  # su Railway può resettarsi dopo deploy (ok)

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
        # su alcuni host filesystem può essere read-only o volatile; non è bloccante
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
    # NON cambia schema (http/https) né host/path ecc.
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))

def transform_mulebuy(url: str) -> str:
    p = urlparse(url)
    if normalize_netloc(p.netloc) != "mulebuy.com":
        return url
    if not p.path.startswith("/product"):
        return url

    q = list(parse_qsl(p.query, keep_blank_values=True))
    # rimuovi qualsiasi ref esistente (anche multipli) e metti SOLO il tuo
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

    # formato voluto: platform, id, ref, poi eventuali altri parametri
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

def rewrite_html_message(html_text: str) -> tuple[str, bool]:
    """
    Riscrive anche i link cliccabili (href="...") mantenendo il formato HTML.
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
    out2, changed2 = rewrite_plain_text(out)
    return out2, (changed or changed2)

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

    rewritten_caption = None
    if buf.caption_html:
        rewritten_caption, _ = rewrite_html_message(buf.caption_html)

    # max 4 foto casuali, video ignorati (non raccolti)
    photos = list(dict.fromkeys(buf.photo_file_ids))
    if len(photos) > MAX_PHOTOS:
        photos = random.sample(photos, k=MAX_PHOTOS)

    if not photos and not rewritten_caption:
        return

    try:
        if photos:
            media = []
            for i, fid in enumerate(photos):
                if i == 0 and rewritten_caption:
                    media.append(InputMediaPhoto(media=fid, caption=rewritten_caption, parse_mode=ParseMode.HTML))
                else:
                    media.append(InputMediaPhoto(media=fid))
            sent_msgs = await context.bot.send_media_group(chat_id=OUTPUT_CHAT_ID_INT, media=media)
            first_id = sent_msgs[0].message_id if sent_msgs else None
        else:
            sent = await context.bot.send_message(
                chat_id=OUTPUT_CHAT_ID_INT,
                text=rewritten_caption,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
            first_id = sent.message_id

        # bollino SOLO per te (reply separata: non inoltrarla)
        if first_id:
            await context.bot.send_message(
                chat_id=OUTPUT_CHAT_ID_INT,
                text="🟣 <b>CONVERTITO</b> (ref aggiornati • max 4 foto • video rimossi)",
                parse_mode=ParseMode.HTML,
                reply_to_message_id=first_id,
            )
    except Exception as e:
        log.exception("Failed to send converted album: %s", e)

# =====================================
# COMMANDS
# =====================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Bot pronto.\n\n"
        "Inoltrami qui in privato i post dei seller.\n"
        "Io li pubblico nel gruppo 'POST PRONTI' con:\n"
        "• link Mulebuy/CNfans col tuo ref\n"
        "• max 4 foto casuali (video ignorati)\n"
        "• data 1 volta al giorno\n\n"
        "Usa /id per ottenere i tuoi ID.",
        disable_web_page_preview=True,
    )

async def cmd_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"🆔 Chat ID: <code>{update.effective_chat.id}</code>\n"
        f"🆔 Your User ID: <code>{update.effective_user.id}</code>",
        parse_mode=ParseMode.HTML
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

    # se non configurato output, avvisa
    if not OUTPUT_CHAT_ID_INT:
        await msg.reply_text("⚠️ OUTPUT_CHAT_ID non impostato. Impostalo nelle variabili di Railway.")
        return

    # ALBUM: raccogli solo foto
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

    # VIDEO singolo: ignoralo (se ha caption, convertila e manda solo testo)
    if msg.video:
        if msg.caption_html:
            rewritten, changed = rewrite_html_message(msg.caption_html)
            if changed:
                await send_daily_header_if_needed(context)
                sent = await context.bot.send_message(
                    chat_id=OUTPUT_CHAT_ID_INT,
                    text=rewritten,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
                await context.bot.send_message(
                    chat_id=OUTPUT_CHAT_ID_INT,
                    text="🟣 <b>CONVERTITO</b> (video ignorato)",
                    parse_mode=ParseMode.HTML,
                    reply_to_message_id=sent.message_id,
                )
        return

    # FOTO singola
    if msg.photo:
        await send_daily_header_if_needed(context)

        cap = msg.caption_html or ""
        rewritten_cap = None
        if cap:
            rewritten_cap, _ = rewrite_html_message(cap)

        sent = await context.bot.send_photo(
            chat_id=OUTPUT_CHAT_ID_INT,
            photo=msg.photo[-1].file_id,
            caption=(rewritten_cap if rewritten_cap else None),
            parse_mode=(ParseMode.HTML if rewritten_cap else None),
        )
        await context.bot.send_message(
            chat_id=OUTPUT_CHAT_ID_INT,
            text="🟣 <b>CONVERTITO</b>",
            parse_mode=ParseMode.HTML,
            reply_to_message_id=sent.message_id,
        )
        return

    # TESTO
    if msg.text_html:
        rewritten, changed = rewrite_html_message(msg.text_html)
        if not changed:
            return  # non sporcare se non c'è nulla da cambiare
        await send_daily_header_if_needed(context)
        sent = await context.bot.send_message(
            chat_id=OUTPUT_CHAT_ID_INT,
            text=rewritten,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )
        await context.bot.send_message(
            chat_id=OUTPUT_CHAT_ID_INT,
            text="🟣 <b>CONVERTITO</b>",
            parse_mode=ParseMode.HTML,
            reply_to_message_id=sent.message_id,
        )

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("id", cmd_id))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle))

    log.info("Bot running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
