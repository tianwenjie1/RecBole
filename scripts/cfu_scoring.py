# -*- coding: utf-8 -*-
# CFU 离线打分：用 clean checkpoint 对训练序列的「最后一个历史 item」算反事实未来效用。
#
# CFU(x_t) = score(x_{t+1} | seq) - score(x_{t+1} | seq_{counterfactual})
#   - mask:   把最后一个历史 item 置 0（padding），保留长度
#   - delete: 把最后一个历史 item 置 0，长度 -1
#
# 输出 csv: row, uid, position, last_item, future_item, item_popularity, is_tail,
#          is_injected_noise, noise_type, CFU_mask, CFU_delete
# 并打印 4 类样本统计 + 存分布图。

import os
import sys
import glob
import csv
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.data.noise_inject import inject_noise
from recbole.utils import init_seed, get_model, init_logger, get_logger


def find_checkpoint(checkpoint_dir, model_name):
    files = glob.glob(os.path.join(checkpoint_dir, f"{model_name}-*.pth"))
    if not files:
        raise FileNotFoundError(f"No checkpoint {model_name}-*.pth in {checkpoint_dir}")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="SASRec")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--noise_type", default="random")
    parser.add_argument("--noise_ratio", type=float, default=0.1)
    parser.add_argument("--noise_seed", type=int, default=2024)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--out", default=None)
    parser.add_argument("--config_files", default=None)
    args = parser.parse_args()

    config_dict = {
        "noise_type": args.noise_type,
        "noise_ratio": args.noise_ratio,
        "noise_seed": args.noise_seed,
        "checkpoint_dir": args.checkpoint_dir,
        "show_progress": False,
        "use_gpu": True,
        "gpu_id": args.gpu_id,
    }
    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=[args.config_files] if args.config_files else None,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    device = config["device"]

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 注入噪声（与训练时同种子），拿到 noise log 标注
    noise_log = inject_noise(train_data._dataset, config, logger=logger)
    corrupted_rows = {e["row"] for e in noise_log}
    logger.info(f"[cfu] corrupted rows = {len(corrupted_rows)}")

    # 加载模型 + clean checkpoint
    model = get_model(config["model"])(config, train_data._dataset).to(device)
    ckpt_path = find_checkpoint(args.checkpoint_dir, args.model)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    logger.info(f"[cfu] loaded checkpoint: {ckpt_path}")

    inter = train_data._dataset.inter_feat
    iid_field = config["ITEM_ID_FIELD"]
    seq_field = iid_field + config["LIST_SUFFIX"]
    len_field = config["ITEM_LIST_LENGTH_FIELD"]
    uid_field = config["USER_ID_FIELD"]
    seq = inter[seq_field]               # [N, max_len]
    length = inter[len_field]            # [N]
    target = inter[iid_field]            # [N]
    uid = inter[uid_field]               # [N]
    N = seq.shape[0]

    # 流行度 / tail
    item_counter = train_data._dataset.item_counter   # Counter: item -> count (train)
    pop_map = {int(k): int(v) for k, v in item_counter.items()}
    sorted_items = sorted(pop_map.items(), key=lambda kv: (kv[1], kv[0]))
    tail_ratio = config["tail_ratio"]
    cut = max(int(len(sorted_items) * tail_ratio), 1)
    tail_set = {it for it, _ in sorted_items[:cut]}

    # 采样
    rng = np.random.RandomState(config["seed"])
    if args.sample_ratio < 1.0:
        n_sample = max(1, int(N * args.sample_ratio))
        rows = rng.choice(N, size=n_sample, replace=False)
        rows.sort()
    else:
        rows = np.arange(N)

    bs = args.batch_size
    results = []
    import torch.nn.functional as F
    item_emb_w = model.item_embedding.weight
    with torch.no_grad():
        for start in range(0, len(rows), bs):
            chunk = rows[start:start + bs]
            s = seq[chunk].to(device)
            ln = length[chunk].to(device)
            tg = target[chunk].to(device)
            B = s.shape[0]
            last_pos = (ln - 1).clamp(min=0)
            rows_idx = torch.arange(B, device=device)

            seq_output = model.forward(s, ln)                 # [B, H]
            orig = (seq_output * item_emb_w[tg]).sum(dim=1)   # [B]
            logits = torch.matmul(seq_output, item_emb_w.transpose(0, 1))
            ce = F.cross_entropy(logits, tg, reduction="none")  # [B]

            # mask: 置 0，保留长度
            mask_s = s.clone()
            mask_s[rows_idx, last_pos] = 0
            m_out = model.forward(mask_s, ln)
            m_score = (m_out * item_emb_w[tg]).sum(dim=1)

            # delete: 置 0，长度 -1
            del_s = mask_s.clone()
            del_ln = (ln - 1).clamp(min=1)
            d_out = model.forward(del_s, del_ln)
            d_score = (d_out * item_emb_w[tg]).sum(dim=1)

            cfu_mask = (orig - m_score).cpu().numpy()
            cfu_del = (orig - d_score).cpu().numpy()
            ce_np = ce.cpu().numpy()
            last_item = s[rows_idx, last_pos].cpu().numpy()
            for i, r in enumerate(chunk):
                li = int(last_item[i])
                results.append({
                    "row": int(r),
                    "uid": int(uid[r]),
                    "position": int(last_pos[i].item()),
                    "last_item": li,
                    "future_item": int(target[r]),
                    "item_popularity": pop_map.get(li, 0),
                    "is_tail": int(li in tail_set),
                    "is_injected_noise": int(int(r) in corrupted_rows),
                    "noise_type": args.noise_type,
                    "CFU_mask": float(cfu_mask[i]),
                    "CFU_delete": float(cfu_del[i]),
                    "orig_loss": float(ce_np[i]),
                })

    out_path = args.out or os.path.join(
        "logs", f"cfu_{args.dataset}_{args.noise_type}_{int(args.noise_ratio*100)}.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"[cfu] wrote {len(results)} rows -> {out_path}")

    # 统计
    import pandas as pd
    df = pd.DataFrame(results)
    df["CFU"] = df["CFU_delete"]   # 主指标用 delete
    groups = {
        "clean_real": df[df["is_injected_noise"] == 0],
        "injected_noise": df[df["is_injected_noise"] == 1],
        "real_tail": df[(df["is_injected_noise"] == 0) & (df["is_tail"] == 1)],
        "real_head": df[(df["is_injected_noise"] == 0) & (df["is_tail"] == 0)],
    }
    print("\n===== CFU separation (CFU_delete) =====")
    print(f"{'group':<18}{'mean':>10}{'median':>10}{'std':>10}{'n':>8}")
    for name, g in groups.items():
        if len(g):
            print(f"{name:<18}{g['CFU'].mean():>10.4f}{g['CFU'].median():>10.4f}"
                  f"{g['CFU'].std():>10.4f}{len(g):>8}")
    # 简易 AUC：能否区分 real vs injected
    from sklearn.metrics import roc_auc_score
    try:
        y = df["is_injected_noise"].values
        # 注入噪声应 CFU 更低 -> 用 -CFU 作为 "noise score"
        auc = roc_auc_score(y, -df["CFU_delete"].values)
        print(f"\nAUC(injected vs real by -CFU_delete) = {auc:.4f}")
    except Exception as e:
        print(f"[auc] skipped: {e}")

    # 分布图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, g in [("clean_real", groups["clean_real"]),
                        ("injected_noise", groups["injected_noise"])]:
            if len(g):
                g["CFU_delete"].clip(-5, 5).hist(bins=50, ax=ax, alpha=0.5, label=name, density=True)
        ax.set_xlabel("CFU_delete")
        ax.set_title(f"{args.dataset} {args.noise_type} {int(args.noise_ratio*100)}%")
        ax.legend()
        plot_path = out_path.replace(".csv", ".png")
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        logger.info(f"[cfu] plot -> {plot_path}")
    except Exception as e:
        logger.info(f"[cfu] plot skipped: {e}")


if __name__ == "__main__":
    main()
