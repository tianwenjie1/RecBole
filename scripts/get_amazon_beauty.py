# -*- coding: utf-8 -*-
# 下载 Amazon Beauty 评分并转 RecBole .inter。
# 用 wget -c 从 hf-mirror.com（国内镜像）下，断点续传；下完再流式解析。
#
# 输出：dataset/amazon-beauty/amazon-beauty.inter
#       字段 user_id:token  item_id:token  rating:float  timestamp:float

import os
import sys
import json
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "dataset", "amazon-beauty")
RAW_PATH = os.path.join(OUT_DIR, "All_Beauty.jsonl")
OUT_PATH = os.path.join(OUT_DIR, "amazon-beauty.inter")

MIRRORS = [
    "https://hf-mirror.com/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl",
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl",
]


def download():
    os.makedirs(OUT_DIR, exist_ok=True)
    # 已下且完整（>100MB）就跳过
    if os.path.exists(RAW_PATH) and os.path.getsize(RAW_PATH) > 100_000_000:
        print(f"[skip] {RAW_PATH} already downloaded ({os.path.getsize(RAW_PATH)} bytes)")
        return
    for url in MIRRORS:
        print(f"[wget] {url}")
        ret = subprocess.call(["wget", "-c", "-O", RAW_PATH, "--timeout=30", "--tries=5", url])
        if ret == 0 and os.path.exists(RAW_PATH) and os.path.getsize(RAW_PATH) > 100_000_000:
            print(f"[ok] downloaded {os.path.getsize(RAW_PATH)} bytes")
            return
        print(f"[fail] wget ret={ret}, size={os.path.getsize(RAW_PATH) if os.path.exists(RAW_PATH) else 0}")
    raise RuntimeError("下载失败。可手动 wget -c 任一镜像 URL 到 " + RAW_PATH)


def convert():
    print(f"[convert] {RAW_PATH} -> {OUT_PATH}")
    n = 0
    with open(RAW_PATH, "r", encoding="utf-8", errors="ignore") as fin, \
         open(OUT_PATH, "w", encoding="utf-8") as fout:
        fout.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            uid = obj.get("user_id")
            iid = obj.get("parent_asin") or obj.get("asin")
            rating = obj.get("rating")
            ts = obj.get("timestamp")
            if uid is None or iid is None or rating is None or ts is None:
                continue
            fout.write(f"{uid}\t{iid}\t{float(rating)}\t{int(ts)}\n")
            n += 1
            if n % 200000 == 0:
                print(f"  ... {n} interactions")
    print(f"[done] wrote {n} interactions -> {OUT_PATH}")


def main():
    convert_only = "--convert-only" in sys.argv
    if not convert_only:
        download()
    convert()


if __name__ == "__main__":
    main()
