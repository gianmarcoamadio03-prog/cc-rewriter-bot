# =========================================================
# TELEGRAM BOT — 100/100 FAST & RELIABLE
# python-telegram-bot v20+
# =========================================================

import os
import asyncio
import random
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from aiolimiter import AsyncLimiter

from telegram import Update, InputMediaPhoto
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut, NetworkError
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    filters,
)

# =========================================================
# CONFIG
# =========================================================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OWNER_ID = int(os.getenv("OWNER_USER_ID"))
OUTPUT_CHAT_ID = int(os.getenv("OUTPUT_CHAT_ID"))

TZ = ZoneInfo("Europe/Rome")

SEND_RATE = 2.5          # invii/sec (stabile)
MEDIA_WAIT = 0.6         # attesa album
BATCH_WAIT = 6.0         # chiusura raffica
PAIR_WINDOW = 10.0       # testo + album = 1 post
MAX_PHOTOS = 4           # foto per album

# =========================================================
# LOG
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("BOT100")

# =========================================================
# UTILS
# =========================================================
def now():
    return datetime.now(TZ).strftime("%H:%M:%S")

def make_id():
    return ''.join(random.choice("ABCDEFGHJKLMNP23456789") for _ in range(4))

def fingerprint(text: str, photos: List[str]) -> str:
    h = hashlib.sha1()
    h.update((text or "").encode())
    for p in photos[:2]:
        h.update(p.encode())
    return h.hexdigest()

# =========================================================
# SEND QUEUE
# =========================================================
QUEUE: asyncio.Queue = asyncio.Queue(maxsize=2000)
LIMITER = AsyncLimiter(3, 1)

async def tg_retry(fn, retries=7):
    for i in range(retries):
        try:
            return await fn()
        except (RetryAfter, TimedOut, NetworkError):
            await asyncio.sleep(0.6 * (2 ** i))
    return await fn()

async def sender_loop(app):
    bot = app.bot
    while True:
        job = await QUEUE.get()
        try:
            async with LIMITER:
                if job["type"] == "album":
                    await tg_retry(lambda: bot.send_media_group(
                        chat_id=OUTPUT_CHAT_ID,
                        media=job["media"]
                    ))
                elif job["type"] == "text":
                    await tg_retry(lambda: bot.send_message(
                        chat_id=OUTPUT_CHAT_ID,
                        text=job["text"],
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True
                    ))
        finally:
            QUEUE.task_done()
            await asyncio.sleep(1 / SEND_RATE)

# =========================================================
# BATCH TRACKER
# =========================================================
@dataclass
class Batch:
    source: str
    emoji: str
    id: str = field(default_factory=make_id)

    total_msgs: int = 0
    texts: int = 0
    photos: int = 0
    albums: int = 0
    posts: int = 0

    last_ts: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    pending_text_ts: Optional[float] = None
    seen_albums: set = field(default_factory=set)
    task: Optional[asyncio.Task] = None

BATCHES: Dict[str, Batch] = {}
PUBLISHED = set()

EMOJIS = ["🟣","🟢","🟠","🔵","🔴","🟡"]

def extract_source(msg):
    fo = msg.forward_origin
    if fo and fo.chat:
        return f"chat:{fo.chat.id}", fo.chat.title or "Chat"
    return "hidden", "Origine nascosta"

async def finalize_batch(key):
    await asyncio.sleep(BATCH_WAIT)
    b = BATCHES.pop(key, None)
    if not b:
        return

    if b.pending_text_ts:
        b.posts += 1

    text = (
        f"{b.emoji} <b>INOLTRO COMPLETATO — #{b.id}</b>\n"
        f"📩 <b>Da:</b> {b.source}\n"
        f"✅ <b>Post:</b> {b.posts}\n"
        f"🧾 <b>Messaggi:</b> {b.total_msgs} | "
        f"💬 {b.texts} | 🖼️ {b.photos} | 🧩 {b.albums}\n"
        f"⏱ {now()}"
    )

    await QUEUE.put({"type": "text", "text": text})

async def batch_touch(msg):
    if not msg.forward_date:
        return

    key, label = extract_source(msg)
    b = BATCHES.get(key)

    if not b:
        b = Batch(source=label, emoji=random.choice(EMOJIS))
        BATCHES[key] = b

    b.total_msgs += 1
    b.last_ts = asyncio.get_event_loop().time()

    if msg.text:
        b.texts += 1
        b.pending_text_ts = b.last_ts

    elif msg.photo:
        b.photos += 1
        if msg.media_group_id:
            if msg.media_group_id not in b.seen_albums:
                b.seen_albums.add(msg.media_group_id)
                b.albums += 1
                b.posts += 1
        else:
            b.posts += 1
        b.pending_text_ts = None

    if b.task and not b.task.done():
        b.task.cancel()
    b.task = asyncio.create_task(finalize_batch(key))

# =========================================================
# COMMANDS
# =========================================================
async def cmd_flush(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        return
    for key in list(BATCHES.keys()):
        await finalize_batch(key)
    await update.message.reply_text("✅ Batch chiuso manualmente.")

# =========================================================
# HANDLER
# =========================================================
async def handle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    if update.effective_chat.type != "private":
        return

    if update.effective_user.id != OWNER_ID:
        return

    await batch_touch(msg)

    # Album
    if msg.media_group_id and msg.photo:
        key = msg.media_group_id
        ctx.chat_data.setdefault(key, []).append(msg.photo[-1].file_id)
        await asyncio.sleep(MEDIA_WAIT)

        photos = ctx.chat_data.pop(key, [])[:MAX_PHOTOS]
        fp = fingerprint("", photos)
        if fp in PUBLISHED:
            return
        PUBLISHED.add(fp)

        media = [InputMediaPhoto(p) for p in photos]
        await QUEUE.put({"type": "album", "media": media})
        return

# =========================================================
# MAIN
# =========================================================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("flush", cmd_flush))
    app.add_handler(MessageHandler(filters.ALL, handle))

    app.post_init = lambda a: asyncio.create_task(sender_loop(a))

    log.info("Bot avviato — FAST MODE")
    app.run_polling()

if __name__ == "__main__":
    main()
