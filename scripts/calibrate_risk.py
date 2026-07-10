# -*- coding: utf-8 -*-
# Stage 2：群体风险受控校准（核心新方法）。
#
# 在 calibration set（注入噪声、有标签）上按组选阈值，保证 deploy 上 tail-FPR≤α、clean-FPR≤β、
# budget≤B，给 Clopper-Pearson 上置信界（conformal risk control）保证。abstention：不确定不修。
#
# 修复规则：repair i iff UCB_i < λ_{g_i}，UCB = score + kappa·std（高不确定性→abstain）。
# 输出 repair csv（同 build_repair 格式，直接 --repair_file=）+ calib json + deploy_validate json。
#
# 用法：
#   python scripts/calibrate_risk.py --cfu_csv logs/cfu_beauty_multistep_early.csv \
#       --score H_del_mean --std H_del_std --alpha 0.005,0.01,0.02 --beta 0.05 \
#       --budget 0.10 --kappa 1.0 --out_dir logs/risk_calib

import os
import csv
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


def cp_upper(k, n, delta):
    """Clopper-Pearson 上置信界（误修率上界），1-delta 置信。"""
    if n <= 0:
        return 1.0
    k = min(max(k, 0), n)
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1 - delta, k + 1, n - k))


def split_cal_deploy(df, cal_frac, seed=0):
    """按 user 分 cal/deploy（避免同一用户行泄漏）。"""
    uids = df["uid"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(uids)
    n_cal = int(len(uids) * cal_frac)
    cal_uids = set(uids[:n_cal].tolist())
    cal_mask = df["uid"].isin(cal_uids).values
    return cal_mask, ~cal_mask


def group_threshold(cal_df, score, alpha, beta, delta, min_n=20):
    """对一组 cal 数据选 λ = max{λ: CP_upper(k_tail,n_tail)≤α ∧ CP_upper(k_clean,n_clean)≤β}。
    k_tail(λ)=#{clean tail: score<λ}（误修真实 tail）；k_clean(λ)=#{clean: score<λ}。"""
    clean = cal_df[cal_df["is_injected_noise"] == 0]
    tail = clean[clean["is_tail"] == 1]
    n_tail = len(tail)
    n_clean = len(clean)
    if n_tail < min_n or n_clean < min_n:
        return None  # 组太小，调用方退维
    s_sorted = np.sort(cal_df[score].values)
    best = -np.inf
    for lam in s_sorted:
        k_tail = int((tail[score].values < lam).sum())
        k_clean = int((clean[score].values < lam).sum())
        if cp_upper(k_tail, n_tail, delta) <= alpha and cp_upper(k_clean, n_clean, delta) <= beta:
            best = lam  # 满足约束，取更大的 λ（修更多噪声）
    return best if best > -np.inf else None


def select_thresholds(df, cal_mask, score, alpha, beta, delta, group_keys, min_n=20):
    """按 group_keys 选 per-group 阈值；小组退维到单 key，再退到全局。"""
    cal = df[cal_mask]
    thresholds = {}      # group_tuple -> λ
    fallback_global = group_threshold(cal, score, alpha, beta, delta, min_n=1)
    # 二维组
    for keys, gcal in cal.groupby(group_keys):
        lam = group_threshold(gcal, score, alpha, beta, delta, min_n)
        if lam is None:
            # 退维：逐单 key
            lam = None
            for gk in group_keys:
                sub = cal[cal[gk] == keys[group_keys.index(gk)]]
                lam = group_threshold(sub, score, alpha, beta, delta, min_n)
                if lam is not None:
                    break
            if lam is None:
                lam = fallback_global
        thresholds[tuple(keys) if isinstance(keys, tuple) else (keys,)] = lam
    return thresholds, fallback_global


def apply_budget(df, deploy_mask, score, thresholds, group_keys, budget, kappa, std_col):
    """budget 约束：若 deploy 预计修复数 > budget·N，二分缩放 c∈(0,1] 使 λ_g*=c·λ_g。"""
    deploy = df[deploy_mask]
    gkeys = deploy[group_keys].apply(tuple, axis=1).values
    lam_arr = np.array([thresholds.get(tuple(gk), 0.0) or 0.0 for gk in gkeys])
    std_arr = deploy[std_col].fillna(0).values if std_col in deploy else np.zeros(len(deploy))
    score_arr = deploy[score].values
    ucb = score_arr + kappa * std_arr

    def n_repair(c):
        return int((ucb < c * lam_arr).sum())

    N = len(deploy)
    cap = int(budget * N)
    if n_repair(1.0) <= cap:
        return 1.0, n_repair(1.0)
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if n_repair(mid) <= cap:
            lo = mid
        else:
            hi = mid
    return lo, n_repair(lo)


def build_repair_csv(df, deploy_mask, score, std_col, thresholds, group_keys, c, kappa, action, pred_col, out_csv):
    """生成 repair csv（row,action,replace_item），repair iff UCB<c·λ_g。"""
    rows_out = []
    deploy = df[deploy_mask]
    gkeys = deploy[group_keys].apply(tuple, axis=1).values
    lam_arr = np.array([thresholds.get(tuple(gk), 0.0) or 0.0 for gk in gkeys])
    std_arr = deploy[std_col].fillna(0).values if std_col in deploy else np.zeros(len(deploy))
    ucb = deploy[score].values + kappa * std_arr
    repair_mask = ucb < c * lam_arr
    for i, (r, rp) in enumerate(zip(deploy["row"].values, repair_mask)):
        if rp:
            rep = int(deploy[pred_col].values[i]) if pred_col in deploy and action == "replace" else ""
            rows_out.append([int(r), action, rep])
        else:
            rows_out.append([int(r), "keep", ""])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "action", "replace_item"])
        w.writerows(rows_out)
    return repair_mask.sum()


def validate_deploy(df, deploy_mask, score, std_col, thresholds, group_keys, c, kappa):
    """deploy 实测指标（有标签但模拟无标签场景）。"""
    deploy = df[deploy_mask]
    gkeys = deploy[group_keys].apply(tuple, axis=1).values
    lam_arr = np.array([thresholds.get(tuple(gk), 0.0) or 0.0 for gk in gkeys])
    std_arr = deploy[std_col].fillna(0).values if std_col in deploy else np.zeros(len(deploy))
    ucb = deploy[score].values + kappa * std_arr
    repair = ucb < c * lam_arr
    is_noise = deploy["is_injected_noise"].values == 1
    is_clean = ~is_noise
    is_tail_clean = is_clean & (deploy["is_tail"].values == 1)
    n_repair = int(repair.sum())
    return {
        "budget_actual": n_repair / len(deploy),
        "repair_count": n_repair,
        "tail_FPR": float((repair & is_tail_clean).sum() / max(is_tail_clean.sum(), 1)),
        "clean_FPR": float((repair & is_clean).sum() / max(is_clean.sum(), 1)),
        "noise_precision": float((repair & is_noise).sum() / max(n_repair, 1)),
        "noise_recall": float((repair & is_noise).sum() / max(is_noise.sum(), 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfu_csv", required=True)
    ap.add_argument("--score", default="H_del_mean")
    ap.add_argument("--std", default="H_del_std")
    ap.add_argument("--pred_col", default="pred_item")
    ap.add_argument("--alpha", default="0.005,0.01,0.02")     # tail-FPR 上限
    ap.add_argument("--beta", default="0.05")                  # clean-FPR 上限
    ap.add_argument("--budget", default="0.10")                # 修复比例上限
    ap.add_argument("--delta", type=float, default=0.1)        # conformal 1-delta=90%
    ap.add_argument("--kappa", default="1.0")                  # abstention
    ap.add_argument("--group_keys", default="pop_bucket,pos_bucket")
    ap.add_argument("--cal_frac", type=float, default=0.5)
    ap.add_argument("--action", default="replace", choices=["mask", "replace"])
    ap.add_argument("--out_dir", default="logs/risk_calib")
    args = ap.parse_args()

    alphas = [float(x) for x in args.alpha.split(",")]
    betas = [float(x) for x in args.beta.split(",")]
    budgets = [float(x) for x in args.budget.split(",")]
    kappas = [float(x) for x in args.kappa.split(",")]
    group_keys = args.group_keys.split(",")

    df = pd.read_csv(args.cfu_csv)
    # 填 NaN score（无效多步行）
    df[args.score] = df[args.score].fillna(df["CFU_delete"])
    if args.std in df:
        df[args.std] = df[args.std].fillna(0.0)
    cal_mask, deploy_mask = split_cal_deploy(df, args.cal_frac)
    print(f"[calibrate] N={len(df)} cal={cal_mask.sum()} deploy={deploy_mask.sum()}")
    print(f"[calibrate] score={args.score} groups={group_keys}")

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for alpha in alphas:
        for beta in betas:
            thresholds, fb = select_thresholds(df, cal_mask, args.score, alpha, beta, args.delta, group_keys)
            for budget in budgets:
                for kappa in kappas:
                    c, n_rep = apply_budget(df, deploy_mask, args.score, thresholds, group_keys, budget, kappa, args.std)
                    tag = f"{args.score}__a{alpha}_b{beta}_B{budget}_k{kappa}"
                    out_csv = os.path.join(args.out_dir, f"repair_{tag}.csv")
                    n_repair = build_repair_csv(df, deploy_mask, args.score, args.std, thresholds, group_keys,
                                                c, kappa, args.action, args.pred_col, out_csv)
                    val = validate_deploy(df, deploy_mask, args.score, args.std, thresholds, group_keys, c, kappa)
                    calib = {
                        "tag": tag, "alpha": alpha, "beta": beta, "budget": budget, "kappa": kappa,
                        "delta": args.delta, "c": c, "thresholds": {str(k): v for k, v in thresholds.items()},
                        "fallback": fb, "deploy_validate": val,
                    }
                    with open(os.path.join(args.out_dir, f"calib_{tag}.json"), "w") as f:
                        json.dump(calib, f, indent=2)
                    summary.append({"tag": tag, "alpha": alpha, "beta": beta, "budget": budget,
                                    "kappa": kappa, **val})
                    print(f"  {tag}: tail_FPR={val['tail_FPR']:.4f} clean_FPR={val['clean_FPR']:.4f} "
                          f"budget={val['budget_actual']:.4f} noise_prec={val['noise_precision']:.4f} "
                          f"noise_recall={val['noise_recall']:.4f}")

    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    # 通过率
    ok = sum(1 for s in summary if s["tail_FPR"] <= s["alpha"] + 1e-9)
    print(f"\n[calibrate] tail-FPR 保证满足: {ok}/{len(summary)} (目标≥90%)")


if __name__ == "__main__":
    main()
