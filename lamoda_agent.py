#!/usr/bin/env python3
"""
Lamoda Shopping Agent
Finds similar items on lamoda.ru based on a photo or product URL.
"""

import anthropic
import requests
from bs4 import BeautifulSoup
import json
import sys
import base64
import re
import os
from pathlib import Path
from urllib.parse import quote, urljoin


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


# ── Step 1: resolve input to an image URL ──────────────────────────────────────

def fetch_product_image_and_name(product_url: str) -> tuple[str, str]:
    """Extract the main product image URL and name from a product page."""
    try:
        resp = requests.get(product_url, headers=HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Could not fetch product page: {e}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try Open Graph first – most reliable across sites
    og_image = soup.find("meta", property="og:image")
    og_title = soup.find("meta", property="og:title")

    image_url = og_image["content"] if og_image else None
    product_name = og_title["content"] if og_title else ""

    # Fallback: twitter:image
    if not image_url:
        tw = soup.find("meta", attrs={"name": "twitter:image"})
        image_url = tw["content"] if tw else None

    # Farfetch-specific: look for the first big product image in JSON-LD
    if not image_url:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                imgs = data.get("image", [])
                if isinstance(imgs, list) and imgs:
                    image_url = imgs[0]
                    break
                elif isinstance(imgs, str):
                    image_url = imgs
                    break
            except Exception:
                pass

    if not image_url:
        raise RuntimeError("Could not find product image on the page.")

    return image_url, product_name.strip()


# ── Step 2: analyze image with Claude Vision ───────────────────────────────────

def analyze_image(client: anthropic.Anthropic, image_input: str) -> dict:
    """
    image_input can be:
      - a local file path
      - an http/https URL
    Returns dict with item_type, color, style, search_query, attributes.
    """
    if image_input.startswith("http"):
        img_content = {
            "type": "image",
            "source": {"type": "url", "url": image_input},
        }
    else:
        path = Path(image_input)
        media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                     ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}
        media_type = media_map.get(path.suffix.lower(), "image/jpeg")
        with open(path, "rb") as f:
            img_data = base64.standard_b64encode(f.read()).decode()
        img_content = {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": img_data},
        }

    prompt = (
        "Analyze this clothing/fashion item. "
        "Return ONLY a valid JSON object (no markdown, no explanation) with these fields:\n"
        "- item_type: clothing type in Russian (e.g. 'платье', 'джинсы', 'куртка', 'кроссовки')\n"
        "- color: dominant color(s) in Russian\n"
        "- style: style description in Russian (e.g. 'повседневный', 'деловой', 'спортивный')\n"
        "- search_query: 2–4 word search phrase in Russian best suited for lamoda.ru search\n"
        "- attributes: list of 3–5 notable attributes (cut, material, pattern, etc.) in Russian"
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": [img_content, {"type": "text", "text": prompt}]}],
    )

    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"Could not parse JSON from Claude: {raw}")


# ── Step 3: search Lamoda ──────────────────────────────────────────────────────

def search_lamoda(query: str) -> tuple[list[dict], str]:
    """
    Searches lamoda.ru for the query.
    Returns (items_list, search_url).
    Each item: {name, brand, price, url, image}.
    """
    search_url = f"https://www.lamoda.ru/catalogsearch/result/?q={quote(query)}"
    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        return [], search_url

    soup = BeautifulSoup(resp.text, "html.parser")
    items = []

    # Lamoda renders products in JSON inside a __NEXT_DATA__ script
    next_data_script = soup.find("script", id="__NEXT_DATA__")
    if next_data_script:
        try:
            next_data = json.loads(next_data_script.string)
            # Navigate the nested structure to find products
            catalog = (
                next_data.get("props", {})
                .get("initialState", {})
                .get("products", {})
                .get("catalog", {})
                .get("products", [])
            )
            for p in catalog[:10]:
                try:
                    brand = p.get("brand", {}).get("name", "") or p.get("brandName", "")
                    title = p.get("title", "") or p.get("name", "")
                    name = f"{brand} — {title}".strip(" —") if brand else title
                    price_info = p.get("prices", {})
                    price = price_info.get("actual", {}).get("amount", "") or price_info.get("basic", {}).get("amount", "")
                    if price:
                        price = f"{price} ₽"
                    slug = p.get("slug", "") or p.get("url", "")
                    url = f"https://www.lamoda.ru{slug}" if slug.startswith("/") else slug
                    images = p.get("images", [])
                    image = images[0].get("url", "") if images else ""
                    if name:
                        items.append({"name": name, "price": price or "—", "url": url, "image": image})
                except Exception:
                    continue
        except Exception:
            pass

    # Fallback: try HTML selectors
    if not items:
        for card in soup.select("div[class*='x-product-card']")[:10]:
            try:
                link = card.select_one("a[href]")
                name_el = card.select_one("[class*='name'], [class*='title']")
                price_el = card.select_one("[class*='price']")
                img_el = card.select_one("img")
                name = name_el.get_text(strip=True) if name_el else (link.get_text(strip=True) if link else "")
                if not name:
                    continue
                href = link["href"] if link else ""
                url = f"https://www.lamoda.ru{href}" if href.startswith("/") else href
                items.append({
                    "name": name,
                    "price": price_el.get_text(strip=True) if price_el else "—",
                    "url": url,
                    "image": img_el.get("src", "") if img_el else "",
                })
            except Exception:
                continue

    return items, search_url


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_agent(image_input: str):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    product_name = ""

    # If it looks like a product page URL (not an image), fetch the image from the page
    is_image_url = bool(re.search(r"\.(jpg|jpeg|png|webp|gif)(\?|$)", image_input, re.I))
    if image_input.startswith("http") and not is_image_url:
        print(f"🌐 Шаг 1/3: Получаю изображение товара со страницы...")
        image_url, product_name = fetch_product_image_and_name(image_input)
        print(f"   Название: {product_name or '(не определено)'}")
        print(f"   Изображение: {image_url[:80]}...")
        image_input = image_url

    print("\n📸 Шаг 2/3: Анализирую вещь с помощью Claude Vision...")
    attrs = analyze_image(client, image_input)
    print(f"   Тип вещи : {attrs.get('item_type', '—')}")
    print(f"   Цвет     : {attrs.get('color', '—')}")
    print(f"   Стиль    : {attrs.get('style', '—')}")
    print(f"   Атрибуты : {', '.join(attrs.get('attributes', []))}")
    query = attrs.get("search_query", attrs.get("item_type", "одежда"))
    print(f"   Запрос   : «{query}»")

    print(f"\n🛍️  Шаг 3/3: Ищу похожее на Lamoda.ru...")
    items, search_url = search_lamoda(query)

    print(f"\n{'─'*60}")
    print(f"🔗 Ссылка на поиск: {search_url}")
    print(f"{'─'*60}")

    if items:
        print(f"\n✨ Найдено похожих товаров: {len(items)}\n")
        for i, item in enumerate(items, 1):
            print(f"  {i:>2}. {item['name']}")
            print(f"       Цена: {item['price']}")
            if item["url"]:
                print(f"       URL : {item['url']}")
            print()
    else:
        print("\n⚠️  Автоматически распарсить карточки не удалось.")
        print("   Откройте ссылку на поиск в браузере — результаты там.")

    return {"attributes": attrs, "search_url": search_url, "items": items}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python lamoda_agent.py <ссылка_на_товар_или_фото>")
        print()
        print("Примеры:")
        print("  python lamoda_agent.py https://www.farfetch.com/item-30125205.aspx")
        print("  python lamoda_agent.py https://example.com/dress.jpg")
        print("  python lamoda_agent.py /path/to/photo.jpg")
        print()
        print("Требуется переменная окружения ANTHROPIC_API_KEY.")
        sys.exit(1)

    run_agent(sys.argv[1])
