#!/usr/bin/env python3
"""Облегчить этап 3 в ноутбуке: загрузка готовых parquet вместо тяжёлых вычислений."""
import json
from pathlib import Path

NB = Path("recommendations.ipynb")

SKIP_PREFIXES = (
    "user_ids_encoded = np.arange",
    'print(f"Похожих пар',
    "candidates_for_train = candidates.copy",
    "def process_events_recs_for_binary_metrics",
    "summary = pd.DataFrame",
)

REPLACEMENTS = [
    (
        'items = pd.read_parquet("items.parquet")\n'
        'events = pd.read_parquet("events.parquet")\n'
        'print(items.shape, events.shape)',
        'items = pd.read_parquet("items.parquet")\n'
        'split_date = pd.Timestamp("2022-12-16")\n'
        'n_train = pl.scan_parquet("events.parquet").filter(pl.col("started_at") < pl.lit(split_date)).select(pl.len()).collect().item()\n'
        'n_test = pl.scan_parquet("events.parquet").filter(pl.col("started_at") >= pl.lit(split_date)).select(pl.len()).collect().item()\n'
        'print(f"items: {items.shape}, events train: {n_train:,}, test: {n_test:,}")',
    ),
    (
        'events["started_at"] = pd.to_datetime(events["started_at"])\n'
        'events.head(3)',
        'events_sample = pl.scan_parquet("events.parquet").head(3).collect().to_pandas()\n'
        'events_sample["started_at"] = pd.to_datetime(events_sample["started_at"])\n'
        'events_sample',
    ),
    (
        'train_test_global_time_split_date = pd.to_datetime("2022-12-16")\n'
        'train_mask = events["started_at"] < train_test_global_time_split_date\n'
        'events_train = events[train_mask].copy()\n'
        'events_test = events[~train_mask].copy()\n'
        'print(f"train: {len(events_train):,}, test: {len(events_test):,}")',
        'train_test_global_time_split_date = pd.Timestamp("2022-12-16")\n'
        'print(f"train: {n_train:,}, test: {n_test:,}")',
    ),
    (
        'print("train:", events_train["started_at"].min(), "—", events_train["started_at"].max())\n'
        'print("test:", events_test["started_at"].min(), "—", events_test["started_at"].max())',
        'train_range = pl.scan_parquet("events.parquet").filter(pl.col("started_at") < pl.lit(split_date)).select(\n'
        '    pl.col("started_at").min(), pl.col("started_at").max()\n'
        ').collect()\n'
        'test_range = pl.scan_parquet("events.parquet").filter(pl.col("started_at") >= pl.lit(split_date)).select(\n'
        '    pl.col("started_at").min(), pl.col("started_at").max()\n'
        ').collect()\n'
        'print("train:", train_range.item(0, 0), "—", train_range.item(0, 1))\n'
        'print("test:", test_range.item(0, 0), "—", test_range.item(0, 1))',
    ),
    (
        "split_date = pl.lit",
        '# Расчёт выполнен scripts/run_stage3.py\n'
        'top_k_pop_items = pd.read_parquet("top_popular.parquet")\n'
        'print(f"Загружено {len(top_k_pop_items)} популярных треков")\n'
        'top_k_pop_items[["rank", "item_id", "plays", "name"]].head(10)',
        lambda t: "pop_items" in t,
    ),
    (
        'top_k_pop_items["score"] = top_k_pop_items["plays"]',
        'top_k_pop_items[["rank", "item_id", "plays", "name"]].head()',
    ),
    (
        "user_encoder = sklearn.preprocessing.LabelEncoder()",
        '# ALS обучена scripts/run_stage3.py\n'
        'personal_als = pd.read_parquet("personal_als.parquet", columns=["user_id", "item_id", "score"])\n'
        'print(personal_als.shape)\n'
        'personal_als.head()',
    ),
    (
        "train_item_ids_enc = events_train",
        'similar_items = pd.read_parquet("similar.parquet")\n'
        'print(f"Похожих пар: {len(similar_items):,}")\n'
        'similar_items.head()',
    ),
    (
        "candidates = personal_als.rename(columns={\"score\": \"als_score\"})",
        'features = ["als_score", "tracks_played_by_user", "item_plays"]\n'
        'print("Признаки:", features)\n'
        'print("Признаки считаются в scripts/run_stage3.py")',
    ),
    (
        "train_pool = Pool(",
        'cb_model = CatBoostClassifier()\n'
        'cb_model.load_model("models/cb_model.cbm")\n'
        'print("CatBoost модель загружена из models/cb_model.cbm")',
    ),
    (
        "candidates_to_rank = candidates_for_train",
        'recommendations = pd.read_parquet("recommendations.parquet")\n'
        'print(recommendations.shape)\n'
        'recommendations.head()',
    ),
    (
        "top_pop_for_eval = []",
        'summary = pd.read_csv("metrics_summary.csv", index_col=0)\n'
        'summary',
    ),
    (
        's3_client.upload_file("top_popular.parquet", S3_BUCKET, "recsys/recommendations/top_popular.parquet")',
        'for fname in ["top_popular", "personal_als", "similar", "recommendations"]:\n'
        '    s3_client.upload_file(f"{fname}.parquet", S3_BUCKET, f"recsys/recommendations/{fname}.parquet")\n'
        'print("Рекомендации загружены в S3: recsys/recommendations/")',
        lambda t: "metrics_top_pop" in t,
    ),
]


def match_replace(text: str) -> str | None:
    for item in REPLACEMENTS:
        prefix, new = item[0], item[1]
        checker = item[2] if len(item) > 2 else None
        if text.startswith(prefix) or prefix in text:
            if checker and not checker(text):
                continue
            return new + "\n"
    for prefix in SKIP_PREFIXES:
        if text.startswith(prefix):
            return "# выполнено scripts/run_stage3.py\n"
    return None


def main() -> None:
    nb = json.loads(NB.read_text())
    changed = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell.get("source", []))
        new_text = match_replace(text)
        if new_text is not None and new_text != text:
            cell["source"] = [new_text]
            cell["outputs"] = []
            cell["execution_count"] = None
            changed += 1
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Patched {changed} cells in {NB}")


if __name__ == "__main__":
    main()
