# @Time   : 2026/07
# CFU-guided input repair：按 repair csv 修复 train 序列里的低效用历史 item。
#
# action:
#   mask    -> 把最后一个历史 item 置 0（padding/zero embedding），保留长度（mask-prediction）
#   replace -> 用 pred_item 替换最后一个历史 item
#   keep    -> 不动
# 修复在输入层，loss 用正常 CE（不加权），保留真实目标 x_{t+1}。

import csv
import torch


def apply_repair(train_dataset, config, logger=None):
    path = config["repair_file"]
    iid_field = config["ITEM_ID_FIELD"]
    seq_field = iid_field + config["LIST_SUFFIX"]
    len_field = config["ITEM_LIST_LENGTH_FIELD"]

    inter_feat = train_dataset.inter_feat
    item_seq = inter_feat[seq_field].clone()
    item_len = inter_feat[len_field]
    N = item_seq.shape[0]

    n_mask = n_replace = 0
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            r = int(row["row"])
            if r < 0 or r >= N:
                continue
            action = row["action"]
            length = int(item_len[r].item())
            if length < 1:
                continue
            pos = length - 1
            if action == "mask":
                item_seq[r, pos] = 0
                n_mask += 1
            elif action == "replace":
                ri = row["replace_item"]
                if ri != "" and ri is not None:
                    item_seq[r, pos] = int(ri)
                    n_replace += 1

    inter_feat[seq_field] = item_seq
    if logger is not None:
        logger.info(f"[input_repair] mask={n_mask} replace={n_replace} (of {N} rows) from {path}")
