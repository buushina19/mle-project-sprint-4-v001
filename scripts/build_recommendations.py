#!/usr/bin/env python3
"""Полный пайплайн части 1 (оптимизирован по памяти, с потоковой записью)."""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

# Немедленный вывод в терминал (иначе при OOM log пустой)
print("Pipeline starting...", flush=True)

import boto3
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse
import sklearn.preprocessing
from catboost import CatBoostClassifier, Pool
from dotenv import load_dotenv
from implicit.als import AlternatingLeastSquares

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
load_dotenv()
os.makedirs("models", exist_ok=True)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
np.random.seed(42)

S3_BUCKET = os.environ["S3_BUCKET_NAME"]
s3_client = boto3.client(
    "s3",
    endpoint_url=os.environ.get("S3_ENDPOINT_URL", "https://storage.yandexcloud.net"),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)

METRICS: dict[str, dict[str, float]] = {}
CHUNK_SIZE = 1_500_000


def log(msg: str) -> None:
    print(msg, flush=True)


def upload(local: str, key: str) -> None:
    s3_client.upload_file(local, S3_BUCKET, key)
    log(f"  uploaded s3://{S3_BUCKET}/{key}")


def write_interactions_clean(valid_tracks: pl.DataFrame, out_path: Path) -> int:
    """Потоковая запись interactions_clean без загрузки всего файла в RAM."""
    if out_path.exists():
        out_path.unlink()

    pf = pq.ParquetFile("data/interactions.parquet")
    writer = None
    total = 0

    for i, batch in enumerate(
        pf.iter_batches(batch_size=CHUNK_SIZE, columns=["user_id", "track_id", "track_seq", "started_at"])
    ):
        chunk = (
            pl.from_arrow(batch)
            .with_columns([
                pl.col("user_id").cast(pl.Int32),
                pl.col("track_id").cast(pl.Int32),
                pl.col("track_seq").cast(pl.Int32),
                pl.col("started_at").cast(pl.Datetime),
            ])
            .filter((pl.col("track_seq") + 2) % 3 == 0)
            .join(valid_tracks, on="track_id", how="inner")
            .drop("track_seq")
        )
        if chunk.height == 0:
            continue

        table = chunk.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        total += chunk.height
        log(f"  interactions chunk {i + 1}: +{chunk.height:,}, total {total:,}")
        del chunk, table, batch
        gc.collect()

    if writer:
        writer.close()
    return total


def stage1_clean() -> tuple[pl.DataFrame, pl.DataFrame]:
    log("=== Stage 1: clean + subsample ===")
    tracks = pl.read_parquet("data/tracks.parquet").with_columns(pl.col("track_id").cast(pl.Int32))
    catalog_names = pl.read_parquet("data/catalog_names.parquet").with_columns(pl.col("id").cast(pl.Int32))
    # Неизвестные id отфильтруются на этапе build_items через join с catalog_names

    clean_path = Path("data/interactions_clean.parquet")
    if not clean_path.exists():
        log("  writing interactions_clean.parquet in chunks (~15-30 min)...")
        valid_tracks = tracks.select("track_id")
        n_events = write_interactions_clean(valid_tracks, clean_path)
        log(f"  wrote {n_events:,} events")
    else:
        n_events = pl.scan_parquet(str(clean_path)).select(pl.len()).collect().item()
        log(f"  interactions_clean.parquet already exists: {n_events:,} events")

    log(f"  tracks: {tracks.shape}")
    return tracks, catalog_names


def build_items(tracks: pl.DataFrame, catalog_names: pl.DataFrame) -> pd.DataFrame:
    log("=== Build items ===")
    cat_tracks = catalog_names.filter(pl.col("type") == "track").select(
        pl.col("id").alias("track_id"), pl.col("name")
    )
    cat_albums = catalog_names.filter(pl.col("type") == "album").select(
        pl.col("id").alias("album_id"), pl.col("name").alias("album_name")
    )
    cat_artists = catalog_names.filter(pl.col("type") == "artist").select(
        pl.col("id").alias("artist_id"), pl.col("name").alias("artist_name")
    )
    cat_genres = catalog_names.filter(pl.col("type") == "genre").select(
        pl.col("id").alias("genre_id"), pl.col("name").alias("genre_name")
    )

    exploded = (
        tracks.select("track_id", "albums", "artists", "genres")
        .explode("albums")
        .explode("artists")
        .explode("genres")
        .rename({"albums": "album_id", "artists": "artist_id", "genres": "genre_id"})
        .drop_nulls(subset=["album_id", "artist_id", "genre_id"])
        .with_columns([
            pl.col("album_id").cast(pl.Int32),
            pl.col("artist_id").cast(pl.Int32),
            pl.col("genre_id").cast(pl.Int32),
        ])
        .join(cat_albums, on="album_id", how="left")
        .join(cat_artists, on="artist_id", how="left")
        .join(cat_genres, on="genre_id", how="left")
    )

    items_pl = (
        exploded.group_by("track_id")
        .agg(
            pl.col("album_name").unique().alias("albums"),
            pl.col("artist_name").unique().alias("artists"),
            pl.col("genre_name").unique().alias("genres"),
        )
        .join(cat_tracks, on="track_id", how="left")
        .rename({"track_id": "item_id"})
    )
    items = items_pl.to_pandas()
    items.to_parquet("items.parquet", index=False)
    log(f"  items: {items.shape}")
    del exploded, items_pl
    gc.collect()
    return items


def prepare_events(items: pd.DataFrame) -> pd.DataFrame:
    log("=== Prepare events ===")
    items_ids = pl.DataFrame({"item_id": items["item_id"].astype(np.int32)})
    (
        pl.scan_parquet("data/interactions_clean.parquet")
        .rename({"track_id": "item_id"})
        .join(items_ids.lazy(), on="item_id", how="inner")
        .sink_parquet("events.parquet")
    )
    events = pd.read_parquet("events.parquet")
    events["started_at"] = pd.to_datetime(events["started_at"])
    log(f"  events: {events.shape}, memory ~{events.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    gc.collect()
    return events


def process_events_recs_for_binary_metrics(events_train, events_test, recs, top_k=5):
    events_test = events_test.copy()
    recs = recs.copy()
    events_test["gt"] = True
    common_users = set(events_test["user_id"]) & set(recs["user_id"])
    events_part = events_test[events_test["user_id"].isin(common_users)].copy()
    recs_part = recs[recs["user_id"].isin(common_users)].copy()
    recs_part = recs_part.sort_values(["user_id", "score"], ascending=[True, False])
    train_items = events_train["item_id"].unique()
    events_part = events_part[events_part["item_id"].isin(train_items)]
    if top_k is not None:
        recs_part = recs_part.groupby("user_id").head(top_k)
    merged = events_part[["user_id", "item_id", "gt"]].merge(
        recs_part[["user_id", "item_id", "score"]], on=["user_id", "item_id"], how="outer"
    )
    merged["gt"] = merged["gt"].fillna(False)
    merged["pr"] = ~merged["score"].isnull()
    merged["tp"] = merged["gt"] & merged["pr"]
    merged["fp"] = ~merged["gt"] & merged["pr"]
    merged["fn"] = merged["gt"] & ~merged["pr"]
    return merged


def compute_cls_metrics(events_recs):
    groupper = events_recs.groupby("user_id")
    precision = (groupper["tp"].sum() / (groupper["tp"].sum() + groupper["fp"].sum())).fillna(0).mean()
    recall = (groupper["tp"].sum() / (groupper["tp"].sum() + groupper["fn"].sum())).fillna(0).mean()
    return float(precision), float(recall)


def compute_novelty(recs, events_train, top_k=5):
    events_train = events_train.copy()
    events_train["played"] = True
    recs = recs.copy().sort_values(["user_id", "score"], ascending=[True, False])
    recs["rank"] = recs.groupby("user_id").cumcount() + 1
    recs = recs.merge(events_train[["user_id", "item_id", "played"]], on=["user_id", "item_id"], how="left")
    recs["played"] = recs["played"].fillna(False)
    return float((1 - recs.query(f"rank <= {top_k}").groupby("user_id")["played"].mean()).mean())


def evaluate(name: str, recs, events_train, events_test, n_items: int) -> None:
    merged = process_events_recs_for_binary_metrics(events_train, events_test, recs, top_k=5)
    precision, recall = compute_cls_metrics(merged)
    coverage = recs["item_id"].nunique() / n_items
    novelty = compute_novelty(recs, events_train, top_k=5)
    METRICS[name] = {"precision": precision, "recall": recall, "coverage": coverage, "novelty": novelty}
    log(f"  {name}: precision={precision:.6f} recall={recall:.6f} coverage={coverage:.6f} novelty={novelty:.6f}")


def build_top_pop_recs(top_k_pop_items: pd.DataFrame, test_users: np.ndarray, top_k: int = 50) -> pd.DataFrame:
    top = top_k_pop_items.head(top_k)[["item_id", "score"]].copy()
    users_df = pd.DataFrame({"user_id": test_users})
    users_df["key"] = 1
    top["key"] = 1
    return users_df.merge(top, on="key").drop(columns="key")


def als_recommend_batched(als_model, user_item_matrix, user_encoder, item_encoder, batch_size=5000, n=50):
    path = Path("personal_als.parquet")
    if path.exists():
        path.unlink()
    writer = None
    n_users = len(user_encoder.classes_)
    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        user_ids_enc = np.arange(start, end)
        raw = als_model.recommend(
            user_ids_enc, user_item_matrix[user_ids_enc], filter_already_liked_items=False, N=n
        )
        batch = pd.DataFrame(
            {"user_id_enc": user_ids_enc, "item_id_enc": raw[0].tolist(), "score": raw[1].tolist()}
        ).explode(["item_id_enc", "score"], ignore_index=True)
        batch["user_id"] = user_encoder.inverse_transform(batch["user_id_enc"].astype(int))
        batch["item_id"] = item_encoder.inverse_transform(batch["item_id_enc"].astype(int))
        batch = batch[["user_id", "item_id", "score"]]
        table = pa.Table.from_pandas(batch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
        log(f"    ALS recs users {end:,}/{n_users:,}")
        del batch, raw, table
        gc.collect()
    if writer:
        writer.close()


def similar_items_batched(als_model, train_item_ids_enc, item_encoder, batch_size=5000, n=11):
    path = Path("similar.parquet")
    if path.exists():
        path.unlink()
    writer = None
    for start in range(0, len(train_item_ids_enc), batch_size):
        chunk = train_item_ids_enc[start : start + batch_size]
        sim_ids, sim_scores = als_model.similar_items(chunk, N=n)
        batch = pd.DataFrame(
            {"item_id_enc": chunk, "sim_item_id_enc": sim_ids.tolist(), "score": sim_scores.tolist()}
        ).explode(["sim_item_id_enc", "score"], ignore_index=True)
        batch["item_id_1"] = item_encoder.inverse_transform(batch["item_id_enc"].astype(int))
        batch["item_id_2"] = item_encoder.inverse_transform(batch["sim_item_id_enc"].astype(int))
        batch = batch.query("item_id_1 != item_id_2")[["item_id_1", "item_id_2", "score"]]
        table = pa.Table.from_pandas(batch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
        log(f"    similar items {min(start + batch_size, len(train_item_ids_enc)):,}/{len(train_item_ids_enc):,}")
        del batch, sim_ids, sim_scores, table
        gc.collect()
    if writer:
        writer.close()
    return pd.read_parquet(path)


def stage3(items: pd.DataFrame, events: pd.DataFrame) -> None:
    log("=== Stage 3 ===")
    split_date = pd.to_datetime("2022-12-16")
    events_train = events[events["started_at"] < split_date].copy()
    events_test = events[events["started_at"] >= split_date].copy()
    log(f"  train: {len(events_train):,}, test: {len(events_test):,}")

    pop_items = (
        events_train.groupby("item_id")
        .agg(plays=("started_at", "count"), users=("user_id", "nunique"))
        .reset_index()
        .sort_values(["plays", "users"], ascending=False)
    )
    top_k_pop_items = pop_items.head(100).reset_index(drop=True)
    top_k_pop_items["rank"] = top_k_pop_items.index + 1
    top_k_pop_items = top_k_pop_items.merge(
        items[["item_id", "name", "genres", "artists", "albums"]], on="item_id", how="inner"
    )
    top_k_pop_items["score"] = top_k_pop_items["plays"] / top_k_pop_items["plays"].sum()
    top_k_pop_items.to_parquet("top_popular.parquet", index=False)

    user_encoder = sklearn.preprocessing.LabelEncoder()
    user_encoder.fit(events["user_id"])
    item_encoder = sklearn.preprocessing.LabelEncoder()
    item_encoder.fit(items["item_id"])

    items = items.copy()
    items["item_id_enc"] = item_encoder.transform(items["item_id"])
    events_train = events_train.merge(items[["item_id", "item_id_enc"]], on="item_id", how="left")
    events_test = events_test.merge(items[["item_id", "item_id_enc"]], on="item_id", how="left")
    events_train["user_id_enc"] = user_encoder.transform(events_train["user_id"])
    events_test["user_id_enc"] = user_encoder.transform(events_test["user_id"])

    events_train["target"] = 1
    user_item_matrix = scipy.sparse.csr_matrix(
        (events_train["target"], (events_train["user_id_enc"], events_train["item_id_enc"])),
        dtype=np.float32,
    )
    del events
    gc.collect()

    log("  training ALS...")
    als_model = AlternatingLeastSquares(factors=50, iterations=15, random_state=42)
    als_model.fit(user_item_matrix)

    log("  personal ALS recommendations...")
    personal_als = als_recommend_batched(als_model, user_item_matrix, user_encoder, item_encoder)

    log("  similar items...")
    similar_items_batched(als_model, events_train["item_id_enc"].unique(), item_encoder)

    log("  ranking model...")
    train_pairs = events_train[["user_id", "item_id"]].drop_duplicates()
    train_pairs["target"] = 1
    candidates = personal_als.rename(columns={"score": "als_score"}).merge(
        train_pairs, on=["user_id", "item_id"], how="left"
    )
    candidates["target"] = candidates["target"].fillna(0).astype(int)
    user_features = events_train.groupby("user_id").agg(tracks_played_by_user=("started_at", "count"))
    item_popularity = events_train.groupby("item_id").agg(item_plays=("started_at", "count"))
    candidates = candidates.merge(user_features, on="user_id", how="left")
    candidates = candidates.merge(item_popularity, on="item_id", how="left")
    candidates[["tracks_played_by_user", "item_plays"]] = candidates[
        ["tracks_played_by_user", "item_plays"]
    ].fillna(0)
    features = ["als_score", "tracks_played_by_user", "item_plays"]

    train_users = np.random.choice(
        candidates["user_id"].unique(), size=min(80_000, candidates["user_id"].nunique()), replace=False
    )
    train_sample = candidates[candidates["user_id"].isin(train_users)]
    log(f"  CatBoost train sample: {len(train_sample):,} rows")
    cb_model = CatBoostClassifier(
        iterations=100, learning_rate=0.1, depth=6, loss_function="Logloss", verbose=25, random_seed=42
    )
    cb_model.fit(Pool(train_sample[features], train_sample["target"]))
    cb_model.save_model("models/cb_model.cbm")

    test_users = events_test["user_id"].unique()
    candidates_test = candidates[candidates["user_id"].isin(test_users)].copy()
    candidates_test["cb_score"] = cb_model.predict_proba(candidates_test[features])[:, 1]
    candidates_test = candidates_test.sort_values(["user_id", "cb_score"], ascending=[True, False])
    candidates_test["rank"] = candidates_test.groupby("user_id").cumcount() + 1
    recommendations = candidates_test.query("rank <= 50")[["user_id", "item_id", "cb_score"]].rename(
        columns={"cb_score": "score"}
    )
    recommendations.to_parquet("recommendations.parquet", index=False)

    log("=== Metrics @5 ===")
    top_pop_recs = build_top_pop_recs(top_k_pop_items, test_users, top_k=50)
    evaluate("Top popular", top_pop_recs, events_train, events_test, len(items))
    evaluate("Personal ALS", personal_als, events_train, events_test, len(items))
    evaluate("Final CatBoost", recommendations, events_train, events_test, len(items))
    pd.DataFrame(METRICS).T.to_csv("metrics_summary.csv")


def main() -> None:
    tracks, catalog_names = stage1_clean()
    items = build_items(tracks, catalog_names)
    del tracks, catalog_names
    gc.collect()
    events = prepare_events(items)

    log("=== Upload data to S3 ===")
    upload("items.parquet", "recsys/data/items.parquet")
    upload("events.parquet", "recsys/data/events.parquet")

    stage3(items, events)

    log("=== Upload recommendations to S3 ===")
    for fname in ["top_popular.parquet", "personal_als.parquet", "similar.parquet", "recommendations.parquet"]:
        upload(fname, f"recsys/recommendations/{fname}")

    log("=== Done ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise
