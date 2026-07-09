# @Time   : 2026/07
# 把 offline CFU 权重挂到 train_dataset.inter_feat 上，使其随 batch 切片流到 calculate_loss。

import csv
import torch


def attach_cfu_weights(train_dataset, config, logger=None):
    """读 cfu_weight_file csv，按 row 对齐写一列权重到 inter_feat。

    csv 需含列: row, weight  (row 为训练样本在 inter_feat 中的行号)
    """
    path = config["cfu_weight_file"]
    field = config["cfu_weight_field"]
    iid_field = config["ITEM_ID_FIELD"]
    seq_field = iid_field + config["LIST_SUFFIX"]
    N = train_dataset.inter_feat[seq_field].shape[0]
    weights = torch.ones(N, dtype=torch.float)
    cnt = 0
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = int(r["row"])
            w = float(r["weight"])
            if 0 <= row < N:
                weights[row] = w
                cnt += 1
    device = train_dataset.inter_feat[seq_field].device
    train_dataset.inter_feat[field] = weights.to(device)
    if logger is not None:
        logger.info(f"[cfu_weight] loaded {cnt}/{N} weights from {path} (field={field})")
