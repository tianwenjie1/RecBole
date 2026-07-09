# -*- coding: utf-8 -*-
# 下载 2014 Amazon Beauty 5-core 评分并转 RecBole .inter。
# 源：HuggingFace milistu/Amazon_Beauty_2014 的 5_core/reviews.parquet（611KB，已 5-core）
# 输出：dataset/amazon-beauty/amazon-beauty.inter
#   字段 user_id:token  item_id:token  rating:float  timestamp:float
# 2014 版稠密（~2万用户、~39万交互），比 2023 All_Beauty 适合做实验。

import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "dataset", "amazon-beauty")
PARQUET_PATH = os.path.join(OUT_DIR, "reviews_5core.parquet")
OUT_PATH = os.path.join(OUT_DIR, "amazon-beauty.inter")

URL = "https://huggingface.co/datasets/milistu/Amazon_Beauty_2014/resolve/main/5_core/reviews.parquet"


def download():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(PARQUET_PATH) and os.path.getsize(PARQUET_PATH) > 100_000:
        print(f"[skip] {PARQUET_PATH} exists ({os.path.getsize(PARQUET_PATH)} bytes)")
        return
    print(f"[get] {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=60).read()
    with open(PARQUET_PATH, "wb") as f:
        f.write(data)
    print(f"[ok] {len(data)} bytes -> {PARQUET_PATH}")


def pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c]
    raise KeyError(f"none of {candidates} in columns {list(df.columns)}")


def convert():
    import pandas as pd
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[parquet] columns={list(df.columns)}, rows={len(df)}")

    uid_col = next(c for c in ["reviewerID", "user_id", "user"] if c in df.columns)
    iid_col = next(c for c in ["asin", "item_id", "parent_asin", "item"] if c in df.columns)
    time_col = next(c for c in ["reviewTime", "timestamp", "unixReviewTime", "time"] if c in df.columns)

    # 该 parquet 是按用户聚合：asin / reviewTime 是列表。爆开成扁平交互。
    df = df.explode([iid_col, time_col]).reset_index(drop=True)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[time_col] = df[time_col].astype("int64") // 10**9  # unix 秒
    out = pd.DataFrame({
        "user_id:token": df[uid_col].astype(str),
        "item_id:token": df[iid_col].astype(str),
        "rating:float": 1.0,
        "timestamp:float": df[time_col].astype(float),
    })
    out.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"[done] wrote {len(out)} interactions -> {OUT_PATH}")
    print(f"  unique users={out['user_id:token'].nunique()}, "
          f"unique items={out['item_id:token'].nunique()}")


def main():
    download()
    convert()


if __name__ == "__main__":
    main()
