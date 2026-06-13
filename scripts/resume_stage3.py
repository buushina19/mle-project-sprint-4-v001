#!/usr/bin/env python3
"""Продолжение этапа 3: CatBoost predict, метрики, S3 (без повторного ALS)."""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from dotenv import load_dotenv

from run_stage3 import (
    METRICS,
    SPLIT,
    add_features,
    iter_als,
    iter_recs,
    load_train_pairs_for_users,
    log,
    upload,
)

load_dotenv()


def valid_parquet(path: Path) -> bool:
    try:
        pq.ParquetFile(path).metadata.num_rows
        return True
    except Exception:
        return False


def main() -> None:
    for path in ("personal_als.parquet", "similar.parquet", "models/cb_model.cbm", "train_pairs.parquet"):
        if not Path(path).exists():
            raise SystemExit(f"Missing {path} — run scripts/run_stage3.py first")

    log("Resume: CatBoost predict + metrics + S3")
    items = pd.read_parquet("items.parquet")
    top = pd.read_parquet("top_popular.parquet")
    catalog_items = set(items["item_id"])

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

    cb = CatBoostClassifier()
    cb.load_model("models/cb_model.cbm")
    feats = ["als_score", "tracks_played_by_user", "item_plays"]

    log("  loading test events...")
    ev_test = (
        pl.scan_parquet("events.parquet")
        .filter(pl.col("started_at") >= SPLIT)
        .select("user_id", "item_id")
        .collect()
        .to_pandas()
    )
    test_u = ev_test["user_id"].unique()

    rp = Path("recommendations.parquet")
    if valid_parquet(rp):
        log(f"  reuse existing recommendations.parquet ({pq.ParquetFile(rp).metadata.num_rows:,} rows)")
    else:
        if rp.exists():
            rp.unlink()
            log("  removed incomplete recommendations.parquet")
        log("CatBoost predict for test users...")
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
        log(f"  recommendations: {pq.ParquetFile(rp).metadata.num_rows:,}")

    mu = test_u[:10_000]
    log("metrics @5...")

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
    main()
