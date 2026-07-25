#!/usr/bin/env python3
"""
Football news bot: finds fresh news via Claude's web_search tool,
formats a Ukrainian Telegram post, finds an image, posts it to a
Telegram draft channel, and keeps a permanent dedup log so the same
story is never posted twice.

Runs as a one-shot script, meant to be triggered every 15 minutes by
a scheduler (GitHub Actions cron in this setup).
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Config (from environment / secrets)
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TG_BOT_TOKEN = os.environ["TG_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TG_CHAT_ID"]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "posted_log.json")
MAX_LOG_KEYS_IN_PROMPT = 500
CAPTION_LIMIT = 900  # keep margin under Telegram's 1024 char caption cap

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ---------------------------------------------------------------------------
# Dedup log
# ---------------------------------------------------------------------------

def load_log():
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_log(log):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def already_posted(log, dedup_key, source_url):
    keys = {entry.get("dedup_key") for entry in log}
    urls = {entry.get("source_url") for entry in log if entry.get("source_url")}
    if dedup_key in keys:
        return True
    if source_url and source_url in urls:
        return True
    return False


# ---------------------------------------------------------------------------
# Claude: search + write posts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Ти — редактор футбольного Telegram-каналу новин. Твоя задача: знайти свіжі, \
ще не опубліковані футбольні новини і підготувати готові пости українською.

ТИПИ НОВИН: трансфери/переходи, зміна тренерів, травми гравців.

ПРІОРИТЕТ ЛІГ: топ-5 (АПЛ, Ла Ліга, Бундесліга, Серія А, Ліга 1), УПЛ + збірна \
України, Ліга чемпіонів/Єврокубки, топ-збірні (ЧС, Євро).

ДЖЕРЕЛА: топ-інсайдери (Fabrizio Romano, David Ornstein, Florian Plettenberg, \
Ben Jacobs та подібні), великі ЗМІ (Sky Sports, ESPN, BBC Sport, Marca, Bild, \
Sport Bild, L'Equipe, Gazzetta, The Athletic), Telegram-канали новин.

ПРАВИЛА ПІДТВЕРДЖЕННЯ:
- Трансфер від топ-інсайдера (Romano/Ornstein/Plettenberg і рівнозначні) — \
досить одного джерела, публікуй одразу.
- Новина про топ-100 футболіста світу АБО про збірну України/її гравців — \
досить одного джерела, навіть якщо це чутка без повного підтвердження.
- Всі інші новини (особливо скандали/конфлікти навколо менш відомих гравців) \
— потрібне підтвердження мінімум 2 незалежних джерел, інакше не бери новину.

ПЕРЕКЛАД: увесь текст посту — українською, незалежно від мови джерела.

ФОРМАТ ПОСТУ (поле post_text), рядки розділені порожнім рядком:
1. ЗАГОЛОВОК: "🗣" + суть новини одним яскравим реченням + " — [Ім'я джерела] \
[емодзі прапора країни джерела]"
2. ОСНОВНИЙ ТЕКСТ: 2-4 речення з контекстом і деталями. Емодзі-прапори клубу/\
збірної/країни всюди, де вони згадуються.
3. "📌 Джерело: [ім'я] ([медіа/статус]) — [рівень довіри: висока / середня / \
потребує підтвердження]"
4. ПИТАННЯ-ЗАКЛИК: провокаційне для трансферів/скандалів, нейтральне/\
співчутливе для травм. Заверши закликом писати думки в коментарях + емодзі 👇.

Без хештегів.

ДЕДУП: тобі дадуть список dedup_key вже опублікованих новин. НЕ включай у \
відповідь нічого, що за суттю збігається з уже опублікованим, навіть якщо \
з'явились нові деталі — це вважається тією самою новиною.

ВІДПОВІДЬ: поверни ТІЛЬКИ JSON-масив (без жодного тексту навколо, без markdown \
пояснень), обгорнутий у ```json ... ```. Кожен елемент масиву:
{
  "dedup_key": "короткий унікальний slug латиницею, напр. transfer-mbappe-real",
  "post_text": "повністю готовий текст посту українською за форматом вище",
  "source_url": "пряме посилання на статтю-першоджерело новини",
  "priority": 1
}
priority — ціле число, 1 = найважливіша новина циклу, більше число = менш \
важлива. Максимум 8 новин за раз. Якщо нових новин немає — поверни порожній \
масив [].
"""


def find_news(existing_keys):
    keys_sample = existing_keys[-MAX_LOG_KEYS_IN_PROMPT:]
    user_prompt = (
        "Знайди свіжі футбольні новини за останні ~30 хвилин (з запасом на "
        "пропущені цикли моніторингу).\n\n"
        "Вже опубліковані dedup_key (не повторюй ці новини за суттю):\n"
        + (", ".join(keys_sample) if keys_sample else "(поки що порожньо)")
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
        messages=[{"role": "user", "content": user_prompt}],
    )

    final_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            final_text += block.text

    match = re.search(r"```json\s*(\[.*?\])\s*```", final_text, re.DOTALL)
    if not match:
        match = re.search(r"(\[.*\])", final_text, re.DOTALL)
    if not match:
        print("No JSON found in Claude's response:", final_text[:500], file=sys.stderr)
        return []

    try:
        items = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e, file=sys.stderr)
        return []

    items.sort(key=lambda x: x.get("priority", 99))
    return items


# ---------------------------------------------------------------------------
# Image extraction (og:image from the source article)
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def extract_og_image(url):
    if not url:
        return None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        html = resp.text
        match = re.search(
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
                html,
                re.IGNORECASE,
            )
        if not match:
            return None
        image_url = match.group(1)
        if re.search(r"\.svg(\?|$)", image_url, re.IGNORECASE):
            return None
        return image_url
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# Telegram sending
# ---------------------------------------------------------------------------

TG_API = f"https://api.telegram.org/bot{TG_BOT_TOKEN}"


def send_photo(photo_url, caption=None):
    payload = {"chat_id": TG_CHAT_ID, "photo": photo_url}
    if caption:
        payload["caption"] = caption
    r = requests.post(f"{TG_API}/sendPhoto", data=payload, timeout=20)
    return r.json()


def send_message(text):
    payload = {"chat_id": TG_CHAT_ID, "text": text}
    r = requests.post(f"{TG_API}/sendMessage", data=payload, timeout=20)
    return r.json()


def publish(item):
    text = item["post_text"]
    image_url = extract_og_image(item.get("source_url"))

    if image_url and len(text) <= CAPTION_LIMIT:
        result = send_photo(image_url, caption=text)
        if result.get("ok"):
            return True
        # bad image, fall back to text-only
        image_url = None

    if image_url:
        photo_result = send_photo(image_url)
        if not photo_result.get("ok"):
            image_url = None

    result = send_message(text)
    return bool(result.get("ok"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log = load_log()
    existing_keys = [entry["dedup_key"] for entry in log if entry.get("dedup_key")]

    items = find_news(existing_keys)
    if not items:
        print("No new news this cycle.")
        return

    for item in items:
        dedup_key = item.get("dedup_key")
        source_url = item.get("source_url")
        post_text = item.get("post_text")

        if not dedup_key or not post_text:
            continue
        if already_posted(log, dedup_key, source_url):
            print(f"Skipping already-posted: {dedup_key}")
            continue

        try:
            ok = publish(item)
        except Exception as e:
            print(f"Failed to publish {dedup_key}: {e}", file=sys.stderr)
            ok = False

        if ok:
            log.append(
                {
                    "dedup_key": dedup_key,
                    "source_url": source_url,
                    "date": datetime.now(timezone.utc).isoformat(),
                }
            )
            save_log(log)
            print(f"Published: {dedup_key}")
        else:
            print(f"Publish failed, not logged: {dedup_key}", file=sys.stderr)

        time.sleep(2)  # be gentle with Telegram's rate limits


if __name__ == "__main__":
    main()
