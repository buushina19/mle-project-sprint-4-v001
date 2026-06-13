#!/usr/bin/env python3
"""Этап 3: рекомендации (оптимизирован под 8 GB RAM)."""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

print("Stage 3 starting...", flush=True)

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

SPLIT = pl.lit(pd.Timestamp("2022-12-16"))
S3_BUCKET = os.environ["S3_BUCKET_NAME"]
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("S3_ENDPOINT_URL", "https://storage.yandexcloud.net"),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)
METRICS: dict[str, dict[str, float]] = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def upload(local: str, key: str) -> None:
    s3.upload_file(local, S3_BUCKET, key)
    log(f"  uploaded s3://{S3_BUCKET}/{key}")


def write_als_recs(als_model, uim, user_enc, item_enc, path="personal_als.parquet", bs=5000, n=50):
    p = Path(path)
    if p.exists():
        p.unlink()
    w = None
    nu = len(user_enc.classes_)
    for s in range(0, nu, bs):
        e = min(s + bs, nu)
        enc = np.arange(s, e)
        raw = als_model.recommend(enc, uim[enc], filter_already_liked_items=False, N=n)
        df = pd.DataFrame({"ue": enc, "ie": raw[0].tolist(), "score": raw[1].tolist()}).explode(
            ["ie", "score"], ignore_index=True
        )
        df["user_id"] = user_enc.inverse_transform(df["ue"].astype(int))
        df["item_id"] = item_enc.inverse_transform(df["ie"].astype(int))
        t = pa.Table.from_pandas(df[["user_id", "item_id", "score"]], preserve_index=False)
        w = w or pq.ParquetWriter(p, t.schema)
        w.write_table(t)
        log(f"    ALS users {e:,}/{nu:,}")
        del df, raw, t
        gc.collect()
    if w:
        w.close()


def write_similar(als_model, item_ids_enc, item_enc, path="similar.parquet", bs=5000):
    p = Path(path)
    if p.exists():
        p.unlink()
    w = None
    for s in range(0, len(item_ids_enc), bs):
        ch = item_ids_enc[s : s + bs]
        sid, sc = als_model.similar_items(ch, N=11)
        df = pd.DataFrame({"ie": ch, "sie": sid.tolist(), "score": sc.tolist()}).explode(
            ["sie", "score"], ignore_index=True
        )
        df["item_id_1"] = item_enc.inverse_transform(df["ie"].astype(int))
        df["item_id_2"] = item_enc.inverse_transform(df["sie"].astype(int))
        df = df.query("item_id_1 != item_id_2")[["item_id_1", "item_id_2", "score"]]
        t = pa.Table.from_pandas(df, preserve_index=False)
        w = w or pq.ParquetWriter(p, t.schema)
        w.write_table(t)
        log(f"    similar {min(s+bs,len(item_ids_enc)):,}/{len(item_ids_enc):,}")
        del df, sid, sc, t
        gc.collect()
    if w:
        w.close()


def iter_als(path="personal_als.parquet", bs=500_000):
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=bs):
        yield b.to_pandas()


def iter_recs(path="recommendations.parquet", bs=500_000):
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=bs):
        yield b.to_pandas()


def load_train_pairs_for_users(users: np.ndarray) -> pd.DataFrame:
    user_set = set(users)
    chunks = []
    for b in pq.ParquetFile("train_pairs.parquet").iter_batches(1_000_000):
        df = b.to_pandas()
        df = df[df["user_id"].isin(user_set)]
        if len(df):
            df["target"] = 1
            chunks.append(df)
    if not chunks:
        return pd.DataFrame(columns=["user_id", "item_id", "target"])
    return pd.concat(chunks, ignore_index=True)


def add_features(cand, train_pairs_df, uf, ip):
    c = cand.rename(columns={"score": "als_score"})
    c = c.merge(train_pairs_df, on=["user_id", "item_id"], how="left")
    c["target"] = c["target"].fillna(0).astype(int)
    c = c.merge(uf, on="user_id", how="left").merge(ip, on="item_id", how="left")
    c[["tracks_played_by_user", "item_plays"]] = c[["tracks_played_by_user", "item_plays"]].fillna(0)
    return c


def main():
    items = pd.read_parquet("items.parquet")
    log(f"items: {items.shape}")

    log("top popular (polars)...")
    pop = (
        pl.scan_parquet("events.parquet")
        .filter(pl.col("started_at") < SPLIT)
        .group_by("item_id")
        .agg(pl.len().alias("plays"), pl.col("user_id").n_unique().alias("users"))
        .sort(["plays", "users"], descending=True)
        .head(100)
        .collect()
        .to_pandas()
    )
    pop["rank"] = np.arange(1, len(pop) + 1)
    top = pop.merge(items[["item_id", "name", "genres", "artists", "albums"]], on="item_id")
    top["score"] = top["plays"] / top["plays"].sum()
    top.to_parquet("top_popular.parquet", index=False)
    log(f"  top_popular: {len(top)} tracks")

    all_users = pl.scan_parquet("events.parquet").select("user_id").unique().collect()["user_id"].to_numpy()
    user_enc = sklearn.preprocessing.LabelEncoder().fit(all_users)
    item_enc = sklearn.preprocessing.LabelEncoder().fit(items["item_id"])

    user_map = pl.DataFrame({
        "user_id": user_enc.classes_,
        "user_id_enc": user_enc.transform(user_enc.classes_),
    })
    item_map = pl.DataFrame({
        "item_id": item_enc.classes_,
        "item_id_enc": item_enc.transform(item_enc.classes_),
    })

    n_train = pl.scan_parquet("events.parquet").filter(pl.col("started_at") < SPLIT).select(pl.len()).collect().item()
    n_test = pl.scan_parquet("events.parquet").filter(pl.col("started_at") >= SPLIT).select(pl.len()).collect().item()
    log(f"  train {n_train:,}, test {n_test:,}")

    log("  user/item features (polars)...")
    uf = (
        pl.scan_parquet("events.parquet")
        .filter(pl.col("started_at") < SPLIT)
        .group_by("user_id")
        .len()
        .collect()
        .to_pandas()
        .rename(columns={"len": "tracks_played_by_user"})
    )
    ip = (
        pl.scan_parquet("events.parquet")
        .filter(pl.col("started_at") < SPLIT)
        .group_by("item_id")
        .len()
        .collect()
        .to_pandas()
        .rename(columns={"len": "item_plays"})
    )

    log("  train pairs -> parquet...")
    tp_path = Path("train_pairs.parquet")
    if not tp_path.exists():
        pl.scan_parquet("events.parquet").filter(pl.col("started_at") < SPLIT).select(
            "user_id", "item_id"
        ).unique().sink_parquet(tp_path)
    else:
        log("    reuse existing train_pairs.parquet")

    log("  building sparse matrix...")
    train_enc = (
        pl.scan_parquet("events.parquet")
        .filter(pl.col("started_at") < SPLIT)
        .select("user_id", "item_id")
        .join(user_map.lazy(), on="user_id")
        .join(item_map.lazy(), on="item_id")
        .collect()
    )
    rows = train_enc["user_id_enc"].to_numpy()
    cols = train_enc["item_id_enc"].to_numpy()
    train_items_enc = train_enc["item_id_enc"].unique().to_numpy()
    del train_enc
    gc.collect()

    uim = scipy.sparse.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols))
    )
    del rows, cols
    gc.collect()

    log("training ALS...")
    als = AlternatingLeastSquares(factors=50, iterations=15, random_state=42)
    als.fit(uim)

    log("personal_als...")
    write_als_recs(als, uim, user_enc, item_enc)
    log("similar...")
    write_similar(als, train_items_enc, item_enc)
    del uim, train_items_enc
    gc.collect()

    feats = ["als_score", "tracks_played_by_user", "item_plays"]
    train_u = set(np.random.choice(all_users, min(50_000, len(all_users)), replace=False))
    log("CatBoost train...")
    chunks = []
    for b in iter_als():
        b = b[b["user_id"].isin(train_u)]
        if len(b):
            tp = load_train_pairs_for_users(b["user_id"].unique())
            chunks.append(add_features(b, tp, uf, ip))
    tr = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    log(f"  train rows: {len(tr):,}")
    cb = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=25, random_seed=42)
    cb.fit(Pool(tr[feats], tr["target"]))
    cb.save_model("models/cb_model.cbm")
    del tr
    gc.collect()

    log("  loading test events...")
    ev_test = (
        pl.scan_parquet("events.parquet")
        .filter(pl.col("started_at") >= SPLIT)
        .select("user_id", "item_id")
        .collect()
        .to_pandas()
    )
    test_u = ev_test["user_id"].unique()
    log("CatBoost predict for test users...")
    rp = Path("recommendations.parquet")
    if rp.exists():
        rp.unlink()
    rw = None
    for b in iter_als():
        b = b[b["user_id"].isin(test_u)]
        if b.empty:
            continue
        tp = load_train_pairs_for_users(b["user_id"].unique())
        c = add_features(b, tp, uf, ip)
        c["cb_score"] = cb.predict_proba(c[feats])[:, 1]
        c = c.sort_values(["user_id", "cb_score"], ascending=[True, False])
        c["rank"] = c.groupby("user_id").cumcount() + 1
        c = c.query("rank<=50")[["user_id", "item_id", "cb_score"]].rename(columns={"cb_score": "score"})
        t = pa.Table.from_pandas(c, preserve_index=False)
        rw = rw or pq.ParquetWriter(rp, t.schema)
        rw.write_table(t)
        del b, c, t, tp
        gc.collect()
    if rw:
        rw.close()

    n_recs = pq.ParquetFile(rp).metadata.num_rows
    log(f"  recommendations: {n_recs:,}")

    mu = test_u[:10_000]
    log("metrics @5...")
    catalog_items = set(item_enc.classes_)

    def proc_metrics(recs_df, top_k=5):
        ev_t = ev_test.copy()
        ev_t["gt"] = True
        cu = set(ev_t["user_id"]) & set(recs_df["user_id"])
        ep = ev_t[ev_t["user_id"].isin(cu)]
        rp_df = recs_df[recs_df["user_id"].isin(cu)].sort_values(["user_id", "score"], ascending=[True, False])
        ep = ep[ep["item_id"].isin(catalog_items)]
        rp_df = rp_df.groupby("user_id").head(top_k)
        m = ep[["user_id", "item_id", "gt"]].merge(rp_df[["user_id", "item_id", "score"]], how="outer")
        m["gt"] = m["gt"].fillna(False)
        m["pr"] = ~m["score"].isnull()
        m["tp"] = m["gt"] & m["pr"]
        m["fp"] = ~m["gt"] & m["pr"]
        m["fn"] = m["gt"] & ~m["pr"]
        return m

    def ev(name, r):
        m = proc_metrics(r)
        g = m.groupby("user_id")
        p = (g["tp"].sum() / (g["tp"].sum() + g["fp"].sum())).fillna(0).mean()
        rc = (g["tp"].sum() / (g["tp"].sum() + g["fn"].sum())).fillna(0).mean()
        cov = r["item_id"].nunique() / len(items)
        tp = load_train_pairs_for_users(r["user_id"].unique())
        tp = tp.rename(columns={"target": "played"})
        rs = r.sort_values(["user_id", "score"], ascending=[True, False])
        rs["rank"] = rs.groupby("user_id").cumcount() + 1
        rs = rs.merge(tp[["user_id", "item_id", "played"]], on=["user_id", "item_id"], how="left")
        rs["played"] = rs["played"].fillna(False)
        nov = (1 - rs.query("rank<=5").groupby("user_id")["played"].mean()).mean()
        METRICS[name] = {"precision": float(p), "recall": float(rc), "coverage": cov, "novelty": float(nov)}
        log(f"  {name}: p={p:.6f} r={rc:.6f} cov={cov:.6f} nov={nov:.6f}")

    top50 = top.head(50)[["item_id", "score"]].copy()
    tdf = pd.DataFrame({"user_id": mu})
    tdf["key"] = 1
    top50["key"] = 1
    ev("Top popular", tdf.merge(top50, on="key").drop(columns="key"))
    als_s = pd.concat([x[x["user_id"].isin(mu)] for x in iter_als(bs=1_000_000)], ignore_index=True)
    ev("Personal ALS", als_s)
    del als_s
    recs_mu = pd.concat([x[x["user_id"].isin(mu)] for x in iter_recs(bs=500_000)], ignore_index=True)
    ev("Final CatBoost", recs_mu)
    del recs_mu
    pd.DataFrame(METRICS).T.to_csv("metrics_summary.csv")
    log(pd.DataFrame(METRICS).T.to_string())

    for f in ["top_popular", "personal_als", "similar", "recommendations"]:
        upload(f"{f}.parquet", f"recsys/recommendations/{f}.parquet")
    log("=== Done ===")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
