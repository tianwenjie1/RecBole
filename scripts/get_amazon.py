# -*- coding: utf-8 -*-
# 下载 2014 Amazon 5-core 评分并转 RecBole .inter（beauty/toys/sports 通用）。
# 源：HuggingFace milistu/Amazon_<Category>_2014 的 5_core/reviews.parquet（已 5-core）
# 用法: python scripts/get_amazon.py --category beauty|toys|sports

import os
import argparse
import urllib.request

CATS = {
    "beauty":  ("milistu/Amazon_Beauty_2014",            "amazon-beauty"),
    "toys":    ("milistu/Amazon_Toys_and_Games_2014",    "amazon-toys-games"),
    "sports":  ("milistu/Amazon_Sports_and_Outdoors_2014", "amazon-sports"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=list(CATS.keys()))
    args = ap.parse_args()
    hf_ds, ds_name = CATS[args.category]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "dataset", ds_name)
    parquet = os.path.join(out_dir, "reviews_5core.parquet")
    out_path = os.path.join(out_dir, f"{ds_name}.inter")
    url = f"https://huggingface.co/datasets/{hf_ds}/resolve/main/5_core/reviews.parquet"

    os.makedirs(out_dir, exist_ok=True)
    if not (os.path.exists(parquet) and os.path.getsize(parquet) > 100000):
        print(f"[get] {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
        with open(parquet, "wb") as f:
            f.write(data)
        print(f"[ok] {len(data)} bytes -> {parquet}")

    import pandas as pd
    df = pd.read_parquet(parquet)
    print(f"[parquet] columns={list(df.columns)}, rows={len(df)}")
    uid_col = next(c for c in ["reviewerID", "user_id", "user"] if c in df.columns)
    iid_col = next(c for c in ["asin", "item_id", "parent_asin", "item"] if c in df.columns)
    time_col = next(c for c in ["reviewTime", "timestamp", "unixReviewTime", "time"] if c in df.columns)
    df = df.explode([iid_col, time_col]).reset_index(drop=True)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce").astype("int64") // 10**9
    out = pd.DataFrame({
        "user_id:token": df[uid_col].astype(str),
        "item_id:token": df[iid_col].astype(str),
        "rating:float": 1.0,
        "timestamp:float": df[time_col].astype(float),
    })
    out.to_csv(out_path, sep="\t", index=False)
    print(f"[done] wrote {len(out)} interactions -> {out_path}")
    print(f"  users={out['user_id:token'].nunique()}, items={out['item_id:token'].nunique()}")


if __name__ == "__main__":
    main()
