"""FastAPI-сервис рекомендаций: офлайн + онлайн с учётом истории пользователя."""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import polars as pl
from fastapi import FastAPI

logger = logging.getLogger("uvicorn.error")
ROOT = Path(__file__).resolve().parent


def dedup_ids(ids: list[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for item_id in ids:
        if item_id not in seen:
            seen.add(item_id)
            result.append(item_id)
    return result


class EventStore:
    """In-memory хранилище последних онлайн-событий пользователя."""

    def __init__(self, max_events_per_user: int = 10) -> None:
        self.events: dict[int, list[int]] = {}
        self.max_events_per_user = max_events_per_user

    def put(self, user_id: int, item_id: int) -> None:
        user_events = self.events.get(user_id, [])
        self.events[user_id] = [item_id] + user_events[: self.max_events_per_user - 1]

    def get(self, user_id: int, k: int) -> list[int]:
        return self.events.get(user_id, [])[:k]


class SimilarItemsStore:
    def __init__(self) -> None:
        self._similar: dict[int, list[tuple[int, float]]] = {}

    def load(self, path: Path) -> None:
        logger.info("Loading similar items from %s", path)
        df = pd.read_parquet(path, columns=["item_id_1", "item_id_2", "score"])
        df = df.sort_values(["item_id_1", "score"], ascending=[True, False])
        grouped = df.groupby("item_id_1")[["item_id_2", "score"]].apply(
            lambda x: list(zip(x["item_id_2"], x["score"]))
        )
        self._similar = grouped.to_dict()
        logger.info("Loaded similar items for %s tracks", len(self._similar))

    def get(self, item_id: int, k: int = 10) -> list[int]:
        pairs = self._similar.get(item_id, [])
        return [item_id_2 for item_id_2, _ in pairs[:k]]


class OfflineRecommendations:
    def __init__(self) -> None:
        self.personal: dict[int, list[int]] = {}
        self.default: list[int] = []
        self.stats = defaultdict(int)

    def load(self, personal_path: Path, default_path: Path) -> None:
        logger.info("Loading offline recommendations")
        personal = pd.read_parquet(personal_path, columns=["user_id", "item_id", "score"])
        personal = personal.sort_values(["user_id", "score"], ascending=[True, False])
        self.personal = personal.groupby("user_id")["item_id"].apply(list).to_dict()

        default = pd.read_parquet(default_path, columns=["item_id"])
        self.default = default["item_id"].tolist()
        logger.info(
            "Loaded personal recs for %s users, default list length %s",
            len(self.personal),
            len(self.default),
        )

    def get(self, user_id: int, k: int = 100) -> list[int]:
        recs = self.personal.get(user_id)
        if recs:
            self.stats["request_personal_count"] += 1
            return recs[:k]
        self.stats["request_default_count"] += 1
        return self.default[:k]


class HistoryStore:
    """Оффлайн-история прослушиваний из events.parquet (lazy lookup по user_id)."""

    def get_seen_items(self, user_id: int) -> set[int]:
        df = (
            pl.scan_parquet(ROOT / "events.parquet")
            .filter(pl.col("user_id") == user_id)
            .select("item_id")
            .collect()
        )
        return set(df["item_id"].to_list())


rec_store = OfflineRecommendations()
similar_store = SimilarItemsStore()
events_store = EventStore()
history_store = HistoryStore()


def filter_seen(item_ids: list[int], seen: set[int]) -> list[int]:
    return [item_id for item_id in item_ids if item_id not in seen]


def blend_recommendations(recs_offline: list[int], recs_online: list[int], k: int) -> list[int]:
    """Оффлайн на чётных позициях (0, 2, …), онлайн — на нечётных (1, 3, …)."""
    blended: list[int] = []
    min_length = min(len(recs_offline), len(recs_online))
    offline_idx = online_idx = 0

    for i in range(2 * min_length):
        if i % 2 == 0:
            blended.append(recs_offline[offline_idx])
            offline_idx += 1
        else:
            blended.append(recs_online[online_idx])
            online_idx += 1

    if len(recs_offline) >= len(recs_online):
        blended.extend(recs_offline[offline_idx:])
    else:
        blended.extend(recs_online[online_idx:])

    return dedup_ids(blended)[:k]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting recommendations service")
    rec_store.load(ROOT / "recommendations.parquet", ROOT / "top_popular.parquet")
    similar_store.load(ROOT / "similar.parquet")
    logger.info("Ready")
    yield
    logger.info("Stats: %s", dict(rec_store.stats))
    logger.info("Stopping")


app = FastAPI(title="recommendations", lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Recommendations service is working"}


@app.post("/events/put")
async def put_event(user_id: int, item_id: int):
    events_store.put(user_id, item_id)
    return {"result": "ok"}


@app.post("/recommendations_offline")
async def recommendations_offline(user_id: int, k: int = 100):
    seen = history_store.get_seen_items(user_id)
    recs = filter_seen(rec_store.get(user_id, k * 2), seen)[:k]
    return {"recs": recs}


@app.post("/recommendations_online")
async def recommendations_online(user_id: int, k: int = 100):
    seen = history_store.get_seen_items(user_id)
    online_events = events_store.get(user_id, k=3)
    seen.update(online_events)

    items: list[int] = []
    for item_id in online_events:
        items.extend(similar_store.get(item_id, k=k))

    recs = filter_seen(dedup_ids(items), seen)[:k]
    return {"recs": recs}


@app.post("/recommendations")
async def recommendations(user_id: int, k: int = 100):
    offline = (await recommendations_offline(user_id, k))["recs"]
    online = (await recommendations_online(user_id, k))["recs"]
    blended = blend_recommendations(offline, online, k)
    return {"recs": blended}
