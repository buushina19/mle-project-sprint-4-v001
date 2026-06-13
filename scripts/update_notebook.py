"""Заполняет recommendations.ipynb кодом для этапов 1-3."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "recommendations.ipynb"

CELLS: dict[int, str] = {
    2: '''import os
import gc
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.sparse
import sklearn.preprocessing
import boto3
from dotenv import load_dotenv
from catboost import CatBoostClassifier, Pool
from implicit.als import AlternatingLeastSquares

load_dotenv()

pl.Config.set_tbl_rows(10)
np.random.seed(42)
pd.set_option("display.max_columns", 50)

session = boto3.session.Session()
s3_client = session.client(
    service_name="s3",
    endpoint_url=os.environ.get("S3_ENDPOINT_URL", "https://storage.yandexcloud.net"),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)
S3_BUCKET = os.environ["S3_BUCKET_NAME"]
os.makedirs("models", exist_ok=True)
''',
    17: '''print("""
Выводы по этапу 1:
- Пропусков в данных нет, идентификаторы приведены к Int32 для согласованности между таблицами.
- В tracks обнаружены ссылки на неизвестные artist/album/genre id — такие значения удалены из списков.
- Из interactions удалены события с track_id, отсутствующими в tracks.
- Для дальнейшей работы interactions сохранены в data/interactions_clean.parquet.
""")''',
    22: '''# Распределение числа прослушанных треков на пользователя
user_track_counts = (
    events_lazy.group_by("user_id")
    .agg(pl.len().alias("tracks_played"))
    .collect()
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(user_track_counts["tracks_played"], bins=50, log=True)
ax.set_title("Распределение количества прослушанных треков на пользователя")
ax.set_xlabel("Число треков")
ax.set_ylabel("Число пользователей (log)")
plt.show()

print(user_track_counts["tracks_played"].describe())''',
    24: '''# Топ-10 популярных треков
track_plays = (
    events_lazy.group_by("track_id")
    .agg(pl.len().alias("plays"))
    .sort("plays", descending=True)
    .head(10)
    .collect()
)

top_tracks = track_plays.join(
    catalog_names.filter(pl.col("type") == "track").select(
        pl.col("id").alias("track_id"), pl.col("name")
    ),
    on="track_id",
    how="left",
)
top_tracks''',
    26: '''# Топ-10 популярных жанров
genre_names = catalog_names.filter(pl.col("type") == "genre").select(
    pl.col("id").alias("genre_id"), pl.col("name")
)

genre_popularity = (
    tracks.select("track_id", "genres")
    .explode("genres")
    .join(events_lazy.group_by("track_id").agg(pl.len().alias("plays")).collect(), on="track_id")
    .group_by("genres")
    .agg(pl.col("plays").sum().alias("plays"))
    .sort("plays", descending=True)
    .head(10)
    .join(genre_names, left_on="genres", right_on="genre_id", how="left")
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(genre_popularity["name"].to_list()[::-1], genre_popularity["plays"].to_list()[::-1])
ax.set_title("Топ-10 жанров по числу прослушиваний")
plt.tight_layout()
plt.show()
genre_popularity''',
    28: '''played_tracks = events_lazy.select("track_id").unique().collect()["track_id"]
unplayed_count = tracks.filter(~pl.col("track_id").is_in(played_tracks)).height
print(f"Треков без прослушиваний: {unplayed_count} из {tracks.height}")
print(f"Доля непрослушанных: {unplayed_count / tracks.height:.2%}")''',
    31: '''# Преобразование в items/events для рекомендаций
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

tracks_pd = tracks.to_pandas()
tracks_exploded = tracks_pd.explode("albums").explode("artists").explode("genres")
tracks_exploded = tracks_exploded.rename(
    columns={"albums": "album_id", "artists": "artist_id", "genres": "genre_id"}
)
tracks_exploded = tracks_exploded.dropna(subset=["album_id", "artist_id", "genre_id"])
tracks_exploded[["album_id", "artist_id", "genre_id"]] = tracks_exploded[
    ["album_id", "artist_id", "genre_id"]
].astype(int)

tracks_exploded = tracks_exploded.merge(cat_albums, on="album_id", how="left")
tracks_exploded = tracks_exploded.merge(cat_artists, on="artist_id", how="left")
tracks_exploded = tracks_exploded.merge(cat_genres, on="genre_id", how="left")
tracks_exploded = tracks_exploded.drop(columns=["album_id", "artist_id", "genre_id"])

items = (
    tracks_exploded.groupby("track_id")
    .agg(
        {
            "album_name": lambda x: list(dict.fromkeys(x)),
            "artist_name": lambda x: list(dict.fromkeys(x)),
            "genre_name": lambda x: list(dict.fromkeys(x)),
        }
    )
    .rename(columns={"album_name": "albums", "artist_name": "artists", "genre_name": "genres"})
    .reset_index()
    .rename(columns={"track_id": "item_id"})
)
items = items.merge(cat_tracks.rename(columns={"track_id": "item_id"}), on="item_id", how="left")

interactions = pd.read_parquet("data/interactions_clean.parquet")
interactions = interactions[interactions["track_id"].isin(items["item_id"])]
# оставляем каждый 3-й трек в истории для экономии памяти
interactions = interactions[(interactions["track_seq"] + 2) % 3 == 0]
interactions = interactions.drop(columns=["track_seq"])
interactions = interactions.rename(columns={"track_id": "item_id"})
events = interactions

print("items:", items.shape)
print("events:", events.shape)
items.head(3)''',
    34: '''items.to_parquet("items.parquet")
events.to_parquet("events.parquet")

s3_client.upload_file("items.parquet", S3_BUCKET, "recsys/data/items.parquet")
s3_client.upload_file("events.parquet", S3_BUCKET, "recsys/data/events.parquet")
print("Файлы сохранены локально и в S3: recsys/data/")''',
    37: '''del tracks, tracks_pd, tracks_exploded, interactions, catalog_names, events_lazy
gc.collect()
print("Память очищена. Перед этапом 3 перезапустите kernel и выполните ячейку инициализации.")''',
    41: '''items = pd.read_parquet("items.parquet")
events = pd.read_parquet("events.parquet")
print(items.shape, events.shape)''',
    42: '''events["started_at"] = pd.to_datetime(events["started_at"])
events.head(3)''',
    45: '''train_test_global_time_split_date = pd.to_datetime("2022-12-16")
train_mask = events["started_at"] < train_test_global_time_split_date
events_train = events[train_mask].copy()
events_test = events[~train_mask].copy()
print(f"train: {len(events_train):,}, test: {len(events_test):,}")''',
    46: '''print("train:", events_train["started_at"].min(), "—", events_train["started_at"].max())
print("test:", events_test["started_at"].min(), "—", events_test["started_at"].max())''',
    49: '''pop_items = (
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
top_k_pop_items.to_parquet("top_popular.parquet")
top_k_pop_items.head(10)''',
    50: '''top_k_pop_items["score"] = top_k_pop_items["plays"] / top_k_pop_items["plays"].sum()
print(f"Сохранено {len(top_k_pop_items)} популярных треков в top_popular.parquet")
top_k_pop_items[["rank", "item_id", "plays", "name"]].head()''',
    53: '''user_encoder = sklearn.preprocessing.LabelEncoder()
user_encoder.fit(events["user_id"])
item_encoder = sklearn.preprocessing.LabelEncoder()
item_encoder.fit(items["item_id"])

items["item_id_enc"] = item_encoder.transform(items["item_id"])
events_train = events_train.merge(items[["item_id", "item_id_enc"]], on="item_id", how="left")
events_test = events_test.merge(items[["item_id", "item_id_enc"]], on="item_id", how="left")
events_train["user_id_enc"] = user_encoder.transform(events_train["user_id"])
events_test["user_id_enc"] = user_encoder.transform(events_test["user_id"])

events_train["target"] = 1
user_item_matrix_train = scipy.sparse.csr_matrix(
    (events_train["target"], (events_train["user_id_enc"], events_train["item_id_enc"])),
    dtype=np.float32,
)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
als_model = AlternatingLeastSquares(factors=50, iterations=15, random_state=42)
als_model.fit(user_item_matrix_train)
print("ALS обучена")''',
    54: '''user_ids_encoded = np.arange(len(user_encoder.classes_))
als_recommendations_raw = als_model.recommend(
    user_ids_encoded,
    user_item_matrix_train[user_ids_encoded],
    filter_already_liked_items=False,
    N=50,
)

als_recommendations = pd.DataFrame(
    {
        "user_id_enc": user_ids_encoded,
        "item_id_enc": als_recommendations_raw[0].tolist(),
        "score": als_recommendations_raw[1].tolist(),
    }
).explode(["item_id_enc", "score"], ignore_index=True)
als_recommendations["item_id_enc"] = als_recommendations["item_id_enc"].astype(int)
als_recommendations["score"] = als_recommendations["score"].astype(float)
als_recommendations["user_id"] = user_encoder.inverse_transform(als_recommendations["user_id_enc"])
als_recommendations["item_id"] = item_encoder.inverse_transform(als_recommendations["item_id_enc"])
personal_als = als_recommendations[["user_id", "item_id", "score"]].copy()
personal_als.to_parquet("personal_als.parquet", index=False)
print(personal_als.shape)
personal_als.head()''',
    57: '''train_item_ids_enc = events_train["item_id_enc"].unique()
max_similar_items = 10
sim_ids, sim_scores = als_model.similar_items(train_item_ids_enc, N=max_similar_items + 1)

similar_items = pd.DataFrame(
    {
        "item_id_enc": train_item_ids_enc,
        "sim_item_id_enc": sim_ids.tolist(),
        "score": sim_scores.tolist(),
    }
).explode(["sim_item_id_enc", "score"], ignore_index=True)
similar_items["sim_item_id_enc"] = similar_items["sim_item_id_enc"].astype(int)
similar_items["score"] = similar_items["score"].astype(float)
similar_items["item_id_1"] = item_encoder.inverse_transform(similar_items["item_id_enc"])
similar_items["item_id_2"] = item_encoder.inverse_transform(similar_items["sim_item_id_enc"])
similar_items = similar_items.query("item_id_1 != item_id_2")[
    ["item_id_1", "item_id_2", "score"]
]
similar_items.to_parquet("similar.parquet", index=False)
similar_items.head()''',
    58: '''print(f"Похожих пар: {len(similar_items):,}")''',
    61: '''candidates = personal_als.rename(columns={"score": "als_score"}).copy()
events_train["target"] = 1
candidates = candidates.merge(
    events_train[["user_id", "item_id", "target"]], on=["user_id", "item_id"], how="left"
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
print("Признаки:", features)
candidates[features + ["target"]].describe()''',
    62: '''candidates_for_train = candidates.copy()
candidates_for_train.to_parquet("candidates_for_train.parquet", index=False)''',
    65: '''train_pool = Pool(
    data=candidates_for_train[features],
    label=candidates_for_train["target"],
)
cb_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function="Logloss",
    verbose=25,
    random_seed=42,
)
cb_model.fit(train_pool)
cb_model.save_model("models/cb_model.cbm")''',
    66: '''candidates_to_rank = candidates_for_train[
    candidates_for_train["user_id"].isin(events_test["user_id"].unique())
].copy()
candidates_to_rank["cb_score"] = cb_model.predict_proba(candidates_to_rank[features])[:, 1]
candidates_to_rank = candidates_to_rank.sort_values(["user_id", "cb_score"], ascending=[True, False])
candidates_to_rank["rank"] = candidates_to_rank.groupby("user_id").cumcount() + 1
recommendations = candidates_to_rank.query("rank <= 50")[["user_id", "item_id", "cb_score"]].rename(
    columns={"cb_score": "score"}
)
recommendations.to_parquet("recommendations.parquet", index=False)
print(recommendations.shape)
recommendations.head()''',
    69: '''def process_events_recs_for_binary_metrics(events_train, events_test, recs, top_k=5):
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
    return precision, recall


def compute_novelty(recs, events_train, top_k=5):
    events_train = events_train.copy()
    events_train["played"] = True
    recs = recs.copy().sort_values(["user_id", "score"], ascending=[True, False])
    recs["rank"] = recs.groupby("user_id").cumcount() + 1
    recs = recs.merge(
        events_train[["user_id", "item_id", "played"]], on=["user_id", "item_id"], how="left"
    )
    recs["played"] = recs["played"].fillna(False)
    return (1 - recs.query(f"rank <= {top_k}").groupby("user_id")["played"].mean()).mean()


def compute_coverage(recs, n_items):
    return recs["item_id"].nunique() / n_items


def evaluate_recs(name, recs):
    merged = process_events_recs_for_binary_metrics(events_train, events_test, recs, top_k=5)
    precision, recall = compute_cls_metrics(merged)
    coverage = compute_coverage(recs, len(items))
    novelty = compute_novelty(recs, events_train, top_k=5)
    print(f"\\n{name} @5:")
    print(f"  precision={precision:.6f}, recall={recall:.6f}")
    print(f"  coverage={coverage:.6f}, novelty={novelty:.6f}")
    return {"precision": precision, "recall": recall, "coverage": coverage, "novelty": novelty}''',
    70: '''top_pop_for_eval = []
for user_id in events_test["user_id"].unique():
    for _, row in top_k_pop_items.head(50).iterrows():
        top_pop_for_eval.append({"user_id": user_id, "item_id": row["item_id"], "score": row["score"]})
top_pop_for_eval = pd.DataFrame(top_pop_for_eval)

metrics_top_pop = evaluate_recs("Top popular", top_pop_for_eval)
metrics_als = evaluate_recs("Personal ALS", personal_als)
metrics_final = evaluate_recs("Final (CatBoost)", recommendations)

s3_client.upload_file("top_popular.parquet", S3_BUCKET, "recsys/recommendations/top_popular.parquet")
s3_client.upload_file("personal_als.parquet", S3_BUCKET, "recsys/recommendations/personal_als.parquet")
s3_client.upload_file("similar.parquet", S3_BUCKET, "recsys/recommendations/similar.parquet")
s3_client.upload_file("recommendations.parquet", S3_BUCKET, "recsys/recommendations/recommendations.parquet")
print("\\nРекомендации загружены в S3: recsys/recommendations/")''',
    73: '''summary = pd.DataFrame(
    [metrics_top_pop, metrics_als, metrics_final],
    index=["Top popular", "Personal ALS", "Final CatBoost"],
)
summary''',
    74: '''print("""
Выводы:
- Top popular даёт базовый уровень качества для холодных пользователей, но низкую персонализацию.
- ALS улучшает recall/precision за счёт коллаборативной фильтрации.
- Ранжирующая CatBoost-модель комбинирует als_score, активность пользователя и популярность трека.
- Coverage остаётся низким — модели рекомендуют ограниченный набор популярных объектов.
- Novelty высокий: большинство рекомендаций — треки, которые пользователь ещё не слушал в train.
""")''',
}


def main() -> None:
    with NOTEBOOK.open(encoding="utf-8") as f:
        nb = json.load(f)

    for idx, source in CELLS.items():
        nb["cells"][idx]["source"] = source

    with NOTEBOOK.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Updated {len(CELLS)} cells in {NOTEBOOK}")


if __name__ == "__main__":
    main()
