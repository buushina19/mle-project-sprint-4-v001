"""Тестирование микросервиса рекомендаций."""

from __future__ import annotations

import logging
import sys

import requests

BASE_URL = "http://127.0.0.1:8000"
HEADERS = {"Content-type": "application/json", "Accept": "text/plain"}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("test_service.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def post(path: str, params: dict) -> dict:
    resp = requests.post(f"{BASE_URL}{path}", headers=HEADERS, params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


def test_user_without_personal_recommendations() -> None:
    """Пользователь без персональных рекомендаций — fallback на top_popular."""
    user_id = 0
    logging.info("=== Test 1: user without personal recommendations (user_id=%s) ===", user_id)
    result = post("/recommendations_offline", {"user_id": user_id, "k": 10})
    logging.info("offline recs: %s", result["recs"])
    assert len(result["recs"]) == 10


def test_user_with_personal_without_online_history() -> None:
    """Пользователь с офлайн-рекомендациями, но без онлайн-истории."""
    user_id = 4
    logging.info("=== Test 2: personal offline only, no online history (user_id=%s) ===", user_id)
    offline = post("/recommendations_offline", {"user_id": user_id, "k": 10})
    online = post("/recommendations_online", {"user_id": user_id, "k": 10})
    blended = post("/recommendations", {"user_id": user_id, "k": 10})
    logging.info("offline recs: %s", offline["recs"])
    logging.info("online recs: %s", online["recs"])
    logging.info("blended recs: %s", blended["recs"])
    assert len(offline["recs"]) == 10
    assert online["recs"] == []
    assert len(blended["recs"]) == 10
    assert blended["recs"] == offline["recs"]


def test_user_with_personal_and_online_history() -> None:
    """Пользователь с офлайн-рекомендациями и онлайн-историей."""
    user_id = 4
    online_items = [6705451, 15458349, 21675009]
    logging.info("=== Test 3: personal + online history (user_id=%s) ===", user_id)

    for item_id in online_items:
        post("/events/put", {"user_id": user_id, "item_id": item_id})
        logging.info("added online event item_id=%s", item_id)

    offline = post("/recommendations_offline", {"user_id": user_id, "k": 10})
    online = post("/recommendations_online", {"user_id": user_id, "k": 10})
    blended = post("/recommendations", {"user_id": user_id, "k": 10})
    logging.info("offline recs: %s", offline["recs"])
    logging.info("online recs: %s", online["recs"])
    logging.info("blended recs: %s", blended["recs"])
    assert len(offline["recs"]) == 10
    assert len(online["recs"]) > 0
    assert len(blended["recs"]) == 10
    assert blended["recs"] != offline["recs"]


def main() -> None:
    setup_logging()
    logging.info("Health check: %s", requests.get(BASE_URL, timeout=30).json())
    test_user_without_personal_recommendations()
    test_user_with_personal_without_online_history()
    test_user_with_personal_and_online_history()
    logging.info("All tests passed")


if __name__ == "__main__":
    main()
