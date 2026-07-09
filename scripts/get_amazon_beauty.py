# -*- coding: utf-8 -*-
# 下载 Amazon Beauty 评分数据并转成 RecBole .inter 格式。
# 源：HuggingFace McAuley-Lab/Amazon-Reviews-2023 (All_Beauty)。
# 输出：dataset/amazon-beauty/amazon-beauty.inter
#       字段 user_id:token  item_id:token  rating:float  timestamp:float
# RecBole 会按 yaml 里的 user/item_inter_num_interval 做 5-core 过滤。

import os
import sys
import gzip
import json
import urllib.request

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "dataset", "amazon-beauty")
OUT_PATH = os.path.join(OUT_DIR, "amazon-beauty.inter")

CANDIDATES = [
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl.gz",
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/0core/rating_only/All_Beauty.jsonl.zst",
]


def open_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=60)


def try_download():
    for url in CANDIDATES:
        try:
            resp = open_url(url)
            print(f"[ok] {url} -> {resp.status}")
            return url, resp
        except Exception as e:
            print(f"[fail] {url} -> {e}")
    raise RuntimeError("所有候选源都下不了，检查网络或换源。")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    url, resp = try_download()
    is_zst = url.endswith(".zst")

    if is_zst:
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(resp)
        f_in = gzip.GzipFile  # placeholder
        # 用 io 包装
        import io
        stream = io.TextIOWrapper(reader, encoding="utf-8")
        lines_iter = stream
    else:
        stream = gzip.GzipFile(fileobj=resp)
        lines_iter = stream

    n = 0
    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        fout.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for raw in lines_iter:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            uid = obj.get("user_id")
            iid = obj.get("asin") or obj.get("parent_asin")
            rating = obj.get("rating")
            ts = obj.get("timestamp")
            if uid is None or iid is None or rating is None or ts is None:
                continue
            fout.write(f"{uid}\t{iid}\t{float(rating)}\t{int(ts)}\n")
            n += 1
    print(f"[done] wrote {n} interactions -> {OUT_PATH}")


if __name__ == "__main__":
    main()
