# -*- coding: utf-8 -*-
# 下载 Amazon Beauty 评分数据并转成 RecBole .inter 格式。
# 源：HuggingFace McAuley-Lab/Amazon-Reviews-2023 (All_Beauty.jsonl, ~326MB)
# 输出：dataset/amazon-beauty/amazon-beauty.inter
#       字段 user_id:token  item_id:token  rating:float  timestamp:float
# RecBole 会按 yaml 里的 user/item_inter_num_interval 做 5-core 过滤。

import os
import json
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "dataset", "amazon-beauty")
OUT_PATH = os.path.join(OUT_DIR, "amazon-beauty.inter")

URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[get] {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=60)
    print(f"[get] status {resp.status}, streaming...")
    n = 0
    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        fout.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        buf = b""
        while True:
            chunk = resp.read(1 << 20)  # 1MB
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
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
                if n % 200000 == 0:
                    print(f"  ... {n} interactions")
    print(f"[done] wrote {n} interactions -> {OUT_PATH}")


if __name__ == "__main__":
    main()
