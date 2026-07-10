# -*- coding: utf-8 -*-
# CFU 离线打分（Stage 1）：多步反事实未来效用 + 群体校准 + 不确定性。
#
# 对每行训练样本 j（prefix 结束于 x_t，target = x_{t+1}），对 horizon k∈{1,3,5}：
#   预测 x_{t+k}。input = prefix(x_a..x_t) + bridge(x_{t+1}..x_{t+k-1})，future_target = x_{t+k}。
#   RecBole 的 data_augmentation 保证同 user 行按 time 排序、target 连续，故
#   x_{t+m} = target[j+m-1]（同 uid）。bridge = target[j..j+k-2]，future = target[j+k-1]。
# 三视图：orig / mask(x_t→0) / del(删 x_t) / replace(x_t→pred_item)。
# CFU_del_Hk = ℓ_cf - ℓ_orig（loss 空间，正=有用）。3 次 dropout inference 估不确定性。
# 群体校准 z-score（pop×pos 分组，median/MAD）。
#
# 输出 csv（向后兼容旧列 + 新列）。打印单步/多步 ROC-AUC/PR-AUC + 分组均值。

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
from recbole.data.noise_inject import inject_noise
from recbole.utils import init_seed, get_model, init_logger
from logging import getLogger


def find_checkpoint(checkpoint_dir, model_name):
    files = glob.glob(os.path.join(checkpoint_dir, f"{model_name}-*.pth"))
    if not files:
        raise FileNotFoundError(f"No checkpoint {model_name}-*.pth in {checkpoint_dir}")
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]


def build_user_row_index(uid_np):
    """uid 已排序，返回 {uid: [row_start..row_end]}。"""
    idx = {}
    cur = None
    start = 0
    for i in range(len(uid_np)):
        u = int(uid_np[i])
        if cur is None:
            cur = u
            start = i
        elif u != cur:
            idx[cur] = np.arange(start, i)
            cur = u
            start = i
    if cur is not None:
        idx[cur] = np.arange(start, len(uid_np))
    return idx


def build_multistep_inputs(seq_np, length_np, target_np, uid_np, horizons, max_len, pred_item_np):
    """构造多步 4 视图输入。返回 flat arrays（M = 有效 (row,k) 数）。

    x_t 在 padded 数组中的列号恒为 max_len - k（k≤5 << max_len，必在窗内）。
    """
    N = len(length_np)
    rows, ks, future = [], [], []
    inp_orig, inp_mask, inp_rep = [], [], []   # 同长 L+k-1
    inp_del = []                                # 长 L+k-2
    lens_orig, lens_del = [], []
    for j in range(N):
        L = int(length_np[j])
        if L < 1:
            continue
        prefix = seq_np[j, :L].tolist()          # [x_a..x_t]
        u = int(uid_np[j])
        for k in horizons:
            if j + k - 1 >= N:
                continue
            # 有效性：j..j+k-1 同 uid（排序下判首尾即可）
            if int(uid_np[j + k - 1]) != u:
                continue
            bridge = target_np[j:j + k - 1].tolist()     # x_{t+1}..x_{t+k-1}（k-1 个）
            fut = int(target_np[j + k - 1])               # x_{t+k}
            logical_orig = prefix + bridge                # 长 L+k-1
            # 左对齐（RecBole item_id_list 是左对齐，forward 在 len-1 处 gather）
            arr_o = np.zeros(max_len, dtype=np.int64)
            arr_m = np.zeros(max_len, dtype=np.int64)
            arr_r = np.zeros(max_len, dtype=np.int64)
            if len(logical_orig) > max_len:
                keep = logical_orig[-max_len:]            # 窗口超长保留最近 max_len
                lo = max_len
            else:
                keep = logical_orig
                lo = len(logical_orig)
            arr_o[:lo] = keep
            arr_m[:lo] = keep
            arr_r[:lo] = keep
            xt_pos = lo - k                                # x_t 在左对齐数组中的位置
            arr_m[xt_pos] = 0
            arr_r[xt_pos] = int(pred_item_np[j])
            # del 长 L+k-2（删 x_t）
            logical_del = prefix[:L - 1] + bridge
            arr_d = np.zeros(max_len, dtype=np.int64)
            if len(logical_del) > max_len:
                arr_d[:max_len] = logical_del[-max_len:]
                ld = max_len
            else:
                arr_d[:len(logical_del)] = logical_del
                ld = len(logical_del)
            inp_orig.append(arr_o); inp_mask.append(arr_m); inp_rep.append(arr_r); inp_del.append(arr_d)
            lens_orig.append(max(lo, 1))      # clamp ≥1 防 gather 越界（L=1 时 del 为空）
            lens_del.append(max(ld, 1))
            rows.append(j); ks.append(k); future.append(fut)
    return (np.array(rows), np.array(ks), np.array(future),
            np.stack(inp_orig), np.stack(inp_mask), np.stack(inp_rep), np.stack(inp_del),
            np.array(lens_orig), np.array(lens_del))


def forward_loss(model, inp, lens, future, item_emb_w, device, bs):
    """返回 (target_score [M], ce_loss [M])（eval 模式）。
    score = seq_output · emb(future)（score-space CFU 用）；ce 供 orig_loss。"""
    import torch.nn.functional as F
    M = inp.shape[0]
    scores = np.empty(M, dtype=np.float64)
    ces = np.empty(M, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for s in range(0, M, bs):
            e = min(s + bs, M)
            si = torch.from_numpy(inp[s:e]).to(device)
            li = torch.from_numpy(lens[s:e]).to(device).long()
            fi = torch.from_numpy(future[s:e]).to(device).long()
            out = model.forward(si, li)
            logits = torch.matmul(out, item_emb_w.transpose(0, 1))
            scores[s:e] = logits[torch.arange(e - s, device=device), fi].cpu().numpy()
            ces[s:e] = F.cross_entropy(logits, fi, reduction="none").cpu().numpy()
    return scores, ces


def forward_loss_dropout(model, inp, lens, future, item_emb_w, device, bs, n_pass, dropout_modules):
    """n_pass 次 dropout inference，返回 (scores [P,M], ces [P,M])。"""
    import torch.nn.functional as F
    M = inp.shape[0]
    all_score = np.empty((n_pass, M), dtype=np.float64)
    all_ce = np.empty((n_pass, M), dtype=np.float64)
    for p in range(n_pass):
        model.eval()
        for m in dropout_modules:
            m.train()
        with torch.no_grad():
            for s in range(0, M, bs):
                e = min(s + bs, M)
                si = torch.from_numpy(inp[s:e]).to(device)
                li = torch.from_numpy(lens[s:e]).to(device).long()
                fi = torch.from_numpy(future[s:e]).to(device).long()
                out = model.forward(si, li)
                logits = torch.matmul(out, item_emb_w.transpose(0, 1))
                all_score[p, s:e] = logits[torch.arange(e - s, device=device), fi].cpu().numpy()
                all_ce[p, s:e] = F.cross_entropy(logits, fi, reduction="none").cpu().numpy()
    return all_score, all_ce


def calibrate_zscore(score, pop_bucket, pos_bucket, clean_mask):
    """pop×pos 分组 z-score（用 clean 交互估 median/MAD）。"""
    z = np.full(len(score), np.nan, dtype=np.float64)
    for (pb, pos) in set(zip(pop_bucket.tolist(), pos_bucket.tolist())):
        g = (pop_bucket == pb) & (pos_bucket == pos)
        ref = g & clean_mask
        if ref.sum() < 5:
            ref = g          # 组太小退到全组
        if ref.sum() < 2:
            z[g] = 0.0
            continue
        med = np.median(score[ref])
        mad = np.median(np.abs(score[ref] - med))
        z[g] = (score[g] - med) / (1.4826 * mad + 1e-6)
    z = np.nan_to_num(z, nan=0.0)
    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="SASRec")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--noise_type", default="random")
    parser.add_argument("--noise_ratio", type=float, default=0.1)
    parser.add_argument("--noise_seed", type=int, default=2024)
    parser.add_argument("--noise_position", default="last")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--early_ckpt", default=None)
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--out", default=None)
    parser.add_argument("--config_files", default=None)
    # Stage 1 新参数
    parser.add_argument("--horizons", default="1,3,5")
    parser.add_argument("--n_dropout", type=int, default=3)
    parser.add_argument("--alpha_weights", default="0.5,0.3,0.2")
    args = parser.parse_args()
    horizons = [int(x) for x in args.horizons.split(",")]
    alpha_w = [float(x) for x in args.alpha_weights.split(",")]
    # 对齐 alpha_w 到 horizons
    if len(alpha_w) < len(horizons):
        alpha_w = alpha_w + [alpha_w[-1]] * (len(horizons) - len(alpha_w))
    alpha_w = np.array(alpha_w[:len(horizons)])
    alpha_w = alpha_w / alpha_w.sum()

    config_dict = {
        "noise_type": args.noise_type, "noise_ratio": args.noise_ratio,
        "noise_seed": args.noise_seed, "noise_position": args.noise_position,
        "checkpoint_dir": args.checkpoint_dir, "show_progress": False,
        "use_gpu": True, "gpu_id": args.gpu_id,
    }
    config = Config(model=args.model, dataset=args.dataset,
                    config_file_list=[args.config_files] if args.config_files else None,
                    config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    device = config["device"]

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    noise_log = inject_noise(train_data._dataset, config, logger=logger)
    corrupted_rows = {e["row"] for e in noise_log}
    logger.info(f"[cfu] corrupted rows = {len(corrupted_rows)}")

    model = get_model(config["model"])(config, train_data._dataset).to(device)
    ckpt_path = args.early_ckpt if args.early_ckpt else find_checkpoint(args.checkpoint_dir, args.model)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    logger.info(f"[cfu] loaded checkpoint: {ckpt_path}")

    inter = train_data._dataset.inter_feat
    iid_field = config["ITEM_ID_FIELD"]
    seq_field = iid_field + config["LIST_SUFFIX"]
    len_field = config["ITEM_LIST_LENGTH_FIELD"]
    uid_field = config["USER_ID_FIELD"]
    seq_np = inter[seq_field].numpy()
    length_np = inter[len_field].numpy()
    target_np = inter[iid_field].numpy()
    uid_np = inter[uid_field].numpy()
    N = seq_np.shape[0]
    max_len = seq_np.shape[1]

    item_counter = train_data._dataset.item_counter
    pop_map = {int(k): int(v) for k, v in item_counter.items()}
    sorted_items = sorted(pop_map.items(), key=lambda kv: (kv[1], kv[0]))
    tail_ratio = config["tail_ratio"]
    cut = max(int(len(sorted_items) * tail_ratio), 1)
    tail_set = {it for it, _ in sorted_items[:cut]}

    # pos_bucket：frac = length / user_maxlen
    user_maxlen = {}
    for i in range(N):
        u = int(uid_np[i])
        if u not in user_maxlen or length_np[i] > user_maxlen[u]:
            user_maxlen[u] = int(length_np[i])
    frac = np.array([length_np[i] / max(user_maxlen[int(uid_np[i])], 1) for i in range(N)])
    pos_bucket = np.where(frac < 0.2, 0, np.where(frac > 0.8, 2, 1))  # 0=early,1=middle,2=recent
    pop_bucket = np.array([0 if int(seq_np[i, int(length_np[i]) - 1]) in tail_set else 1 for i in range(N)])  # 0=tail,1=head

    # pred_item（单步 del forward 的 argmax，供 replace 用）
    logger.info("[cfu] computing pred_item (single-step del)...")
    bs = args.batch_size
    item_emb_w = model.item_embedding.weight
    pred_item_np = np.zeros(N, dtype=np.int64)
    import torch.nn.functional as F
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s + bs, N)
            si = torch.from_numpy(seq_np[s:e]).to(device)
            li = torch.from_numpy((length_np[s:e] - 1).clip(min=1)).to(device).long()
            out = model.forward(si, li)
            dl = torch.matmul(out, item_emb_w.transpose(0, 1))
            dl[:, 0] = -float("inf")
            pred_item_np[s:e] = dl.argmax(dim=1).cpu().numpy()

    # 多步构造
    logger.info(f"[cfu] building multistep inputs (horizons={horizons})...")
    rows, ks, future, inp_o, inp_m, inp_r, inp_d, lens_o, lens_d = build_multistep_inputs(
        seq_np, length_np, target_np, uid_np, horizons, max_len, pred_item_np)
    M = len(rows)
    logger.info(f"[cfu] multistep pairs = {M} (avg {M/N:.2f} horizons/row)")

    # CFU 均值用 eval（无 dropout，干净，与 Part1 一致）；dropout 只用于不确定性 std
    dropout_modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    logger.info(f"[cfu] forward 4 views (eval) + {args.n_dropout} dropout (orig/mask for std)...")
    sc_o, ce_o = forward_loss(model, inp_o, lens_o, future, item_emb_w, device, bs)
    sc_m, _ = forward_loss(model, inp_m, lens_o, future, item_emb_w, device, bs)
    sc_r, _ = forward_loss(model, inp_r, lens_o, future, item_emb_w, device, bs)
    sc_d, ce_d = forward_loss(model, inp_d, lens_d, future, item_emb_w, device, bs)

    # CFU = score_orig - score_cf（score 空间，正=有用）
    cfu_del = sc_o - sc_d
    cfu_mask = sc_o - sc_m
    cfu_rep = sc_o - sc_r

    # 不确定性：orig+mask 的 dropout pass，算 per-pass H_mask 的 std
    sc_o_drop, _ = forward_loss_dropout(model, inp_o, lens_o, future, item_emb_w, device, bs, args.n_dropout, dropout_modules)
    sc_m_drop, _ = forward_loss_dropout(model, inp_m, lens_o, future, item_emb_w, device, bs, args.n_dropout, dropout_modules)
    Hmask_pass = sc_o_drop - sc_m_drop   # [P, M]

    # 按 (row, k) 聚合到每行
    row_to_idx = {}
    for idx, r in enumerate(rows):
        row_to_idx.setdefault(int(r), {})[int(ks[idx])] = idx

    # 组装每行结果
    results = []
    for j in range(N):
        L = int(length_np[j])
        if L < 1:
            continue
        li = int(seq_np[j, L - 1])
        rec = {
            "row": j, "uid": int(uid_np[j]), "item_length": L,
            "last_item": li, "future_item": int(target_np[j]),
            "item_popularity": pop_map.get(li, 0), "is_tail": int(li in tail_set),
            "pop_bucket": int(pop_bucket[j]), "pos_bucket": int(pos_bucket[j]),
            "frac": float(frac[j]),
            "is_injected_noise": int(j in corrupted_rows), "noise_type": args.noise_type,
            "noise_position": args.noise_position,
            "pred_item": int(pred_item_np[j]),
        }
        hmap = row_to_idx.get(j, {})
        # 单步（H1）向后兼容列
        i1 = hmap.get(1)
        rec["CFU_mask"] = float(cfu_mask[i1]) if i1 is not None else 0.0
        rec["CFU_delete"] = float(cfu_del[i1]) if i1 is not None else 0.0
        rec["orig_loss"] = float(ce_o[i1]) if i1 is not None else 0.0
        # 多步
        for ki, k in enumerate(horizons):
            idx_k = hmap.get(k)
            rec[f"CFU_del_H{k}"] = float(cfu_del[idx_k]) if idx_k is not None else float("nan")
            rec[f"CFU_mask_H{k}"] = float(cfu_mask[idx_k]) if idx_k is not None else float("nan")
            rec[f"CFU_rep_H{k}"] = float(cfu_rep[idx_k]) if idx_k is not None else float("nan")
        # H_del = Σ α_k CFU_del_Hk（只对有效 horizon 重归一化，跳过 NaN）
        vals = [rec[f"CFU_del_H{k}"] for k in horizons]
        num = 0.0; den = 0.0
        for i, v in enumerate(vals):
            if not np.isnan(v):
                num += alpha_w[i] * v
                den += alpha_w[i]
        rec["H_del_mean"] = float(num / den) if den > 0 else 0.0
        # H_del 不确定性：用 mask 视图各 dropout pass 的 H_mask(score-space) 做 std
        if args.n_dropout > 1 and i1 is not None:
            pass_H = []
            for p in range(args.n_dropout):
                hp = 0.0; sw = 0.0
                for ki, k in enumerate(horizons):
                    idx_k = hmap.get(k)
                    if idx_k is None:
                        continue
                    hp += alpha_w[ki] * Hmask_pass[p, idx_k]
                    sw += alpha_w[ki]
                if sw > 0:
                    pass_H.append(hp / sw)
            rec["H_del_std"] = float(np.std(pass_H, ddof=1)) if len(pass_H) > 1 else 0.0
        else:
            rec["H_del_std"] = 0.0
        results.append(rec)

    # 群体校准 z-score
    import pandas as pd
    df = pd.DataFrame(results)
    clean = df["is_injected_noise"].values == 0
    df["z_CFU_del"] = calibrate_zscore(df["CFU_delete"].values, df["pop_bucket"].values, df["pos_bucket"].values, clean)
    df["z_H_del"] = calibrate_zscore(df["H_del_mean"].values, df["pop_bucket"].values, df["pos_bucket"].values, clean)

    out_path = args.out or os.path.join(
        "logs", f"cfu_{args.dataset}_{args.noise_type}_{int(args.noise_ratio*100)}_{args.noise_position}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"[cfu] wrote {len(df)} rows -> {out_path}")

    # 统计：单步 vs 多步 vs 校准
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = df["is_injected_noise"].values
    groups = {
        "clean_real": df[df["is_injected_noise"] == 0],
        "injected_noise": df[df["is_injected_noise"] == 1],
        "real_tail": df[(df["is_injected_noise"] == 0) & (df["is_tail"] == 1)],
        "real_head": df[(df["is_injected_noise"] == 0) & (df["is_tail"] == 0)],
    }
    print(f"\n===== CFU separation ({args.noise_position}) =====")
    print(f"{'group':<16}{'CFU_del':>10}{'H_del':>10}{'z_H_del':>10}{'H_std':>10}{'n':>8}")
    for name, g in groups.items():
        if len(g):
            print(f"{name:<16}{g['CFU_delete'].mean():>10.4f}{g['H_del_mean'].mean():>10.4f}"
                  f"{g['z_H_del'].mean():>10.4f}{g['H_del_std'].mean():>10.4f}{len(g):>8}")
    for col, label in [("CFU_delete", "single-step"), ("H_del_mean", "multi-step"), ("z_H_del", "calibrated")]:
        try:
            auc = roc_auc_score(y, -df[col].values)
            pr = average_precision_score(y, -df[col].values)
            print(f"{label:<12} ROC-AUC={auc:.4f}  PR-AUC={pr:.4f}")
        except Exception as e:
            print(f"{label}: {e}")
    # 覆盖率
    for k in horizons:
        cov = df[f"CFU_del_H{k}"].notna().mean()
        print(f"H={k} coverage={cov:.3f}")
    # 各位置 ROC（若多位置）
    print("\n-- per pos_bucket ROC (z_H_del) --")
    for pb, name in [(0, "early-ish"), (1, "middle"), (2, "recent")]:
        sub = df[df["pos_bucket"] == pb]
        if sub["is_injected_noise"].sum() > 0 and (sub["is_injected_noise"] == 0).sum() > 0:
            try:
                a = roc_auc_score(sub["is_injected_noise"], -sub["z_H_del"])
                print(f"  pos={name}: ROC={a:.4f} n={len(sub)}")
            except Exception:
                pass

    # 分布图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, g in [("clean_real", groups["clean_real"]), ("injected_noise", groups["injected_noise"])]:
            if len(g):
                g["H_del_mean"].clip(-5, 5).hist(bins=50, ax=ax, alpha=0.5, label=name, density=True)
        ax.set_xlabel("H_del"); ax.legend()
        plt.savefig(out_path.replace(".csv", ".png"), dpi=120, bbox_inches="tight")
        logger.info(f"[cfu] plot -> {out_path.replace('.csv', '.png')}")
    except Exception as e:
        logger.info(f"[cfu] plot skipped: {e}")


if __name__ == "__main__":
    main()
