# @Time   : 2026/07
# 噪声注入模块：在训练序列上注入可控噪声，valid/test 不受影响。
#
# 设计说明（与 CFU 一致）：
#   RecBole 的滑动窗口让每行训练样本 = (历史序列 item_id_list, 目标 item_id)。
#   本模块对选中行的「最后一个历史 item」(即 item_id_list[row, item_length-1]) 做替换。
#   这样 CFU 每行直接评估「紧邻目标的上一个交互」对未来目标的反事实效用，
#   注入噪声行的 last-item 即被污染交互，与 CFU 打分位置严格对齐。
#
# noise_type:
#   none              -> 不注入
#   random            -> 均匀随机替换
#   popularity        -> 按训练集流行度分布采样替换（热门曝光噪声）
#   context_mismatch  -> 从其他用户的序列里采样替换（上下文不匹配）

import os
import csv
import numpy as np
import torch


def _build_pop_distribution(item_counter, item_num):
    """返回流行度归一化概率向量，索引为 item id。"""
    prob = np.zeros(item_num, dtype=np.float64)
    for iid, cnt in item_counter.items():
        iid = int(iid)
        if 0 < iid < item_num:
            prob[iid] = cnt
    s = prob.sum()
    if s <= 0:
        prob = np.ones(item_num, dtype=np.float64) / item_num
    else:
        prob = prob / s
    return prob


def inject_noise(train_dataset, config, logger=None):
    """原地给 train_dataset.inter_feat 注入噪声，返回 noise log (list[dict])。

    Args:
        train_dataset: data_preparation 返回的 train_data._dataset
        config: Config 对象，需含 noise_type / noise_ratio / noise_seed
        logger: 可选 logger
    Returns:
        noise_log: [{'uid','row','position','orig_item','new_item','noise_type'}, ...]
    """
    noise_type = config["noise_type"]
    noise_ratio = float(config["noise_ratio"])
    seed = int(config["noise_seed"])
    noise_position = config["noise_position"] if "noise_position" in config else "last"

    rng = np.random.RandomState(seed)
    noise_log = []

    if noise_type in (None, "none", "") or noise_ratio <= 0:
        return noise_log

    iid_field = config["ITEM_ID_FIELD"]
    seq_field = iid_field + config["LIST_SUFFIX"]      # item_id_list
    len_field = config["ITEM_LIST_LENGTH_FIELD"]        # item_length
    uid_field = config["USER_ID_FIELD"]

    inter_feat = train_dataset.inter_feat
    item_seq = inter_feat[seq_field]                    # [N, max_len] tensor
    item_len = inter_feat[len_field]                    # [N] tensor
    uids = inter_feat[uid_field]                        # [N] tensor
    targets = inter_feat[iid_field]                     # [N] target item

    N, max_len = item_seq.shape
    item_num = int(train_dataset.num(iid_field))

    # 按位置 band 选行：frac = item_length / 该 user 最大 item_length
    len_np = item_len.numpy()
    uid_np = uids.numpy()
    user_maxlen = {}
    for i in range(N):
        u = int(uid_np[i])
        if u not in user_maxlen or len_np[i] > user_maxlen[u]:
            user_maxlen[u] = int(len_np[i])
    frac = np.array([len_np[i] / max(user_maxlen[int(uid_np[i])], 1) for i in range(N)])
    if noise_position == "last":
        # 每 user 仅最大 item_length 那行（紧邻 target）
        seen = {}
        band_rows = []
        for i in range(N):
            u = int(uid_np[i])
            if len_np[i] == user_maxlen[u] and u not in seen:
                seen[u] = True
                band_rows.append(i)
        band = np.array(band_rows, dtype=int)
    elif noise_position == "recent":
        band = np.where(frac > 0.8)[0]
    elif noise_position == "middle":
        band = np.where((frac >= 0.4) & (frac <= 0.6))[0]
    elif noise_position == "early":
        band = np.where(frac < 0.2)[0]
    else:  # uniform
        band = np.arange(N)
    if len(band) == 0:
        band = np.arange(N)

    # 选中要污染的行（noise_ratio 比例的总行数，从 band 内采）
    n_rows = int(round(N * noise_ratio))
    n_rows = max(1, min(n_rows, len(band)))
    row_idx = rng.choice(band, size=n_rows, replace=False)

    # 采样替换 item
    if noise_type == "random":
        new_items = rng.randint(1, item_num, size=n_rows)  # 0 是 padding，跳过
    elif noise_type == "popularity":
        prob = _build_pop_distribution(train_dataset.item_counter, item_num)
        new_items = rng.choice(item_num, size=n_rows, p=prob)
        new_items = np.where(new_items == 0, 1, new_items)  # 避开 padding
    elif noise_type == "context_mismatch":
        # 从其他行的历史里随机抽一个 item 作为替换（上下文不匹配的代理）
        all_items = item_seq.flatten().numpy()
        all_items = all_items[all_items > 0]
        new_items = rng.choice(all_items, size=n_rows, replace=True)
    else:
        raise NotImplementedError(f"noise_type={noise_type} not supported")

    item_seq_np = item_seq.clone()
    for k, r in enumerate(row_idx):
        r = int(r)
        length = int(item_len[r].item())
        if length < 1:
            continue
        pos = length - 1                          # 最后一个历史位置
        orig = int(item_seq_np[r, pos].item())
        new = int(new_items[k])
        if new == orig:
            new = (new % (item_num - 1)) + 1      # 强制不同
        item_seq_np[r, pos] = new
        noise_log.append({
            "uid": int(uids[r].item()),
            "row": r,
            "position": pos,
            "orig_item": orig,
            "new_item": new,
            "target_item": int(targets[r].item()),
            "noise_type": noise_type,
            "noise_position": noise_position,
        })

    # 写回
    inter_feat[seq_field] = item_seq_np

    # 保存 noise log
    log_dir = config["noise_log_dir"] if "noise_log_dir" in config else "logs"
    os.makedirs(log_dir, exist_ok=True)
    dataset_name = config["dataset"]
    log_path = os.path.join(
        log_dir, f"noise_{dataset_name}_{noise_type}_{int(noise_ratio*100)}_{noise_position}_seed{seed}.csv"
    )
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "uid", "row", "position", "orig_item", "new_item", "target_item", "noise_type", "noise_position"
        ])
        writer.writeheader()
        writer.writerows(noise_log)

    if logger is not None:
        logger.info(
            f"[noise_inject] type={noise_type} ratio={noise_ratio} position={noise_position} "
            f"rows={N} band={len(band)} corrupted={len(noise_log)} log={log_path}"
        )
    return noise_log
