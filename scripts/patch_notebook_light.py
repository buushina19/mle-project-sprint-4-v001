#!/usr/bin/env python3
"""Сделать ноутбук безопасным для 8 GB: не грузить большие parquet целиком."""
import json
from pathlib import Path

NB = Path("recommendations.ipynb")

CLEAN_CELL = '''from pathlib import Path
import pyarrow.parquet as pq

if Path("data/interactions_clean.parquet").exists():
    n = pq.ParquetFile("data/interactions_clean.parquet").metadata.num_rows
    print(f"Используем готовый data/interactions_clean.parquet, событий: {n:,}")
    events_lazy = pl.scan_parquet("data/interactions_clean.parquet")
else:
    artist_ids = catalog_names.filter(pl.col("type") == "artist")["id"].implode()
    album_ids = catalog_names.filter(pl.col("type") == "album")["id"].implode()
    genre_ids = catalog_names.filter(pl.col("type") == "genre")["id"].implode()
    for col, valid_ids in [("artists", artist_ids), ("albums", album_ids), ("genres", genre_ids)]:
        n = tracks.filter(
            pl.col(col).list.eval(pl.element().filter(~pl.element().is_in(valid_ids))).list.len() > 0
        ).height
        print(f"{col}: треков с неизвестными id = {n}")
    unknown_events = (
        events_lazy.select("track_id").unique()
        .join(tracks.select("track_id").lazy(), on="track_id", how="anti")
        .select(pl.len()).collect().item()
    )
    print(f"неизвестных track_id в events: {unknown_events}")
    tracks = tracks.with_columns([
        pl.col("artists").list.eval(pl.element().filter(pl.element().is_in(artist_ids))),
        pl.col("albums").list.eval(pl.element().filter(pl.element().is_in(album_ids))),
        pl.col("genres").list.eval(pl.element().filter(pl.element().is_in(genre_ids))),
    ])
    print("tracks после очистки:", tracks.shape)
    print("Сохраняем interactions_clean.parquet...")
    (
        events_lazy.select([
            pl.col("user_id").cast(pl.Int32),
            pl.col("track_id").cast(pl.Int32),
            pl.col("track_seq").cast(pl.Int32),
            pl.col("started_at").cast(pl.Datetime),
        ])
        .join(tracks.select("track_id").lazy(), on="track_id", how="inner")
        .sink_parquet("data/interactions_clean.parquet")
    )
    events_lazy = pl.scan_parquet("data/interactions_clean.parquet")
    print("events после очистки:", events_lazy.select(pl.len()).collect().item())
'''

ITEMS_CELL = '''import pyarrow.parquet as pq

if Path("items.parquet").exists() and Path("events.parquet").exists():
    items = pd.read_parquet("items.parquet")
    n_events = pq.ParquetFile("events.parquet").metadata.num_rows
    print("items:", items.shape)
    print("events:", n_events)
    items.head(3)
else:
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
    interactions = interactions[(interactions["track_seq"] + 2) % 3 == 0]
    interactions = interactions.drop(columns=["track_seq"])
    interactions = interactions.rename(columns={"track_id": "item_id"})
    events = interactions
    print("items:", items.shape)
    print("events:", events.shape)
    items.head(3)
'''

SAVE_CELL = '''if Path("items.parquet").exists() and Path("events.parquet").exists():
    print("items.parquet и events.parquet уже сохранены локально и в S3 (recsys/data/)")
else:
    items.to_parquet("items.parquet")
    events.to_parquet("events.parquet")
    s3_client.upload_file("items.parquet", S3_BUCKET, "recsys/data/items.parquet")
    s3_client.upload_file("events.parquet", S3_BUCKET, "recsys/data/events.parquet")
    print("Файлы сохранены локально и в S3: recsys/data/")
'''

GC_CELL = '''for _v in ["tracks", "tracks_pd", "tracks_exploded", "interactions", "catalog_names", "events_lazy", "events"]:
    if _v in globals():
        del globals()[_v]
gc.collect()
print("Память очищена. Этап 3 использует готовые parquet-файлы.")
'''

PERSONAL_ALS = '''import pyarrow.parquet as pq
pf = pq.ParquetFile("personal_als.parquet")
print(f"personal_als: {pf.metadata.num_rows:,} rows")
personal_als = pq.read_table("personal_als.parquet").slice(0, 5).to_pandas()
personal_als
'''

RECOMMENDATIONS = '''pf = pq.ParquetFile("recommendations.parquet")
print(f"recommendations: {pf.metadata.num_rows:,} rows")
recommendations = pq.read_table("recommendations.parquet").slice(0, 5).to_pandas()
recommendations
'''

SIMILAR = '''pf = pq.ParquetFile("similar.parquet")
print(f"Похожих пар: {pf.metadata.num_rows:,}")
similar_items = pq.read_table("similar.parquet").slice(0, 5).to_pandas()
similar_items
'''


def set_cell(cell, text: str) -> None:
    cell["source"] = [text + "\n"]
    cell["outputs"] = []
    cell["execution_count"] = None


def main() -> None:
    nb = json.loads(NB.read_text())
    changed = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell.get("source", []))
        if "sink_parquet(\"data/interactions_clean.parquet\")" in text and "artist_ids" in text:
            set_cell(cell, CLEAN_CELL)
            changed += 1
        elif text.startswith("# Преобразование в items/events"):
            set_cell(cell, ITEMS_CELL)
            changed += 1
        elif text.startswith("items.to_parquet(\"items.parquet\")"):
            set_cell(cell, SAVE_CELL)
            changed += 1
        elif text.startswith("del tracks, tracks_pd"):
            set_cell(cell, GC_CELL)
            changed += 1
        elif "personal_als.parquet" in text and "pd.read_parquet" in text:
            set_cell(cell, PERSONAL_ALS)
            changed += 1
        elif "recommendations.parquet" in text and "pd.read_parquet" in text:
            set_cell(cell, RECOMMENDATIONS)
            changed += 1
        elif text.startswith("similar_items = pd.read_parquet"):
            set_cell(cell, SIMILAR)
            changed += 1
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"Patched {changed} cells in {NB}")


if __name__ == "__main__":
    main()
