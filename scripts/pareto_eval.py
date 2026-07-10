# -*- coding: utf-8 -*-
# Stage 3：matched-budget / matched-risk 评测。
#
# matched-risk：对每个方法的 score 套【同一个】conformal 风险控制 wrapper（calibrate_risk 逻辑），
#   在固定 tail-FPR 下选阈值，隔离「检测分数质量」vs「风险控制策略」。
#   方法 score：
#     ours        : H_del_mean（多步校准）
#     cfu_single  : CFU_delete（单步，= CARD-PER 的 future-contribution 信号）
#     loss_reweight: -orig_loss（高 loss=noise，取负使 low=noise）
#     pad_like    : CFU_delete - gamma*pop_norm（popularity-aware，tail 更宽松）
# matched-budget：每方法固定修复 5/10/15%（percentile 选）。
#
# 输出：每方法×约束 的 repair csv（供训练）+ matched-risk 检测指标表 + Pareto 数据。
# 训练后用 collect_results 聚合 NDCG。

import os
import csv
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

# 复用 calibrate_risk 的函数
from calibrate_risk import split_cal_deploy, cp_upper, group_threshold, select_thresholds, apply_budget, build_repair_csv, validate_deploy


# 方法：(name, score_col, std_col, kappa)。proposed = cfu_single + 风险控制 + abstention
METHODS = [
    ("loss_reweight_rc", "_neg_orig_loss", None, 0.0),
    ("pad_like_rc", "_pad_like", None, 0.0),
    ("cfu_single_rc", "CFU_delete", None, 0.0),
    ("proposed", "CFU_delete", "H_del_std", 1.0),   # 单步 CFU + 风险控制 + 弃权
]


def prepare_scores(df, gamma=0.5):
    """补派生 score 列。"""
    if "_neg_orig_loss" not in df:
        df["_neg_orig_loss"] = -df["orig_loss"].fillna(0)
    if "_pad_like" not in df:
        pop = df.get("item_popularity", pd.Series(np.zeros(len(df))))
        pop_norm = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
        df["_pad_like"] = df["CFU_delete"].fillna(0) - gamma * pop_norm
    return df


def matched_risk(df, cal_mask, deploy_mask, methods, alphas, beta, delta, group_keys, budget, out_dir, kappa=0.0):
    """对每方法×alpha 选 conformal 阈值，输出 repair csv + validate。每方法自带 kappa/std。"""
    df["_zero"] = 0.0
    rows = []
    for mname, score_col, std_col, m_kappa in methods:
        std_col_use = std_col if (m_kappa > 0 and std_col and std_col in df) else "_zero"
        for alpha in alphas:
            thresholds, fb = select_thresholds(df, cal_mask, score_col, alpha, beta, delta, group_keys)
            if fb is None:
                print(f"  {mname} a={alpha}: 阈值选取失败（组太小？）跳过")
                continue
            c, _ = apply_budget(df, deploy_mask, score_col, thresholds, group_keys, budget, m_kappa, std_col_use)
            tag = f"{mname}__a{alpha}"
            out_csv = os.path.join(out_dir, f"repair_{tag}.csv")
            all_mask = np.ones(len(df), dtype=bool)
            n_repair = build_repair_csv(df, all_mask, score_col, std_col_use, thresholds, group_keys,
                                        c, m_kappa, "replace", "pred_item", out_csv)
            val = validate_deploy(df, deploy_mask, score_col, std_col_use, thresholds, group_keys, c, m_kappa)
            rows.append({"method": mname, "alpha": alpha, "score": score_col, "kappa": m_kappa, **val})
            with open(os.path.join(out_dir, f"calib_{tag}.json"), "w") as f:
                json.dump({"method": mname, "alpha": alpha, "c": c, "kappa": m_kappa,
                           "deploy_validate": val,
                           "thresholds": {str(k): v for k, v in thresholds.items()}}, f, indent=2)
            print(f"  {mname:18s} a={alpha}: tail_FPR={val['tail_FPR']:.4f} noise_prec={val['noise_precision']:.4f} "
                  f"noise_recall={val['noise_recall']:.4f} budget={val['budget_actual']:.4f} abstain_kappa={m_kappa}")
    return rows


def matched_budget(df, cal_mask, deploy_mask, methods, budgets, group_keys, out_dir):
    """每方法固定修复 budget 比例（per-group percentile 等价全局 percentile）。输出 repair csv。"""
    rows = []
    for mname, score_col, _, _ in methods:
        s = df[score_col].fillna(0).values
        for budget in budgets:
            thr = np.quantile(s, budget)   # 修 bottom budget 比例
            tag = f"{mname}__B{budget}"
            out_csv = os.path.join(out_dir, f"repair_{tag}.csv")
            repair = s < thr
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["row", "action", "replace_item"])
                for i, r in enumerate(df["row"].values):
                    if repair[i]:
                        w.writerow([int(r), "replace", int(df["pred_item"].values[i])])
                    else:
                        w.writerow([int(r), "keep", ""])
            # validate
            is_noise = df["is_injected_noise"].values == 1
            is_clean = ~is_noise
            is_tail_clean = is_clean & (df["is_tail"].values == 1)
            n_rep = int(repair.sum())
            val = {
                "budget_actual": n_rep / len(df), "repair_count": n_rep,
                "tail_FPR": float((repair & is_tail_clean).sum() / max(is_tail_clean.sum(), 1)),
                "noise_precision": float((repair & is_noise).sum() / max(n_rep, 1)),
                "noise_recall": float((repair & is_noise).sum() / max(is_noise.sum(), 1)),
            }
            rows.append({"method": mname, "budget": budget, **val})
            print(f"  {mname:14s} B={budget}: tail_FPR={val['tail_FPR']:.4f} noise_prec={val['noise_precision']:.4f} "
                  f"noise_recall={val['noise_recall']:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfu_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alphas", default="0.005,0.01,0.02")
    ap.add_argument("--budgets", default="0.05,0.10,0.15")
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--budget_cap", type=float, default=0.15)
    ap.add_argument("--group_keys", default="pop_bucket,pos_bucket")
    ap.add_argument("--cal_frac", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=0.0, help="0=公平比检测分数质量(无弃权); >0 时 ours 用 H_del_std 弃权")
    args = ap.parse_args()

    df = pd.read_csv(args.cfu_csv)
    df = prepare_scores(df, args.gamma)
    cal_mask, deploy_mask = split_cal_deploy(df, args.cal_frac)
    alphas = [float(x) for x in args.alphas.split(",")]
    budgets = [float(x) for x in args.budgets.split(",")]
    group_keys = args.group_keys.split(",")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== matched-risk (tail-FPR 受控, kappa={args.kappa}) ===  cal={cal_mask.sum()} deploy={deploy_mask.sum()}")
    mr = matched_risk(df, cal_mask, deploy_mask, METHODS, alphas, args.beta, args.delta, group_keys, args.budget_cap, args.out_dir, args.kappa)
    print(f"\n=== matched-budget (固定修复比例) ===")
    mb = matched_budget(df, cal_mask, deploy_mask, METHODS, budgets, group_keys, args.out_dir)

    pd.DataFrame(mr).to_csv(os.path.join(args.out_dir, "matched_risk.csv"), index=False)
    pd.DataFrame(mb).to_csv(os.path.join(args.out_dir, "matched_budget.csv"), index=False)

    # matched-risk 关键对比：同 alpha 下各方法 noise_recall（检测力）
    print("\n=== matched-risk 检测力对比（同 tail-FPR 下 noise_recall，越高越好）===")
    mrt = pd.DataFrame(mr).pivot(index="method", columns="alpha", values="noise_recall")
    print(mrt.to_string())
    print("\n=== matched-risk tail-FPR 是否达标（应≤alpha）===")
    mrf = pd.DataFrame(mr).pivot(index="method", columns="alpha", values="tail_FPR")
    print(mrf.to_string())
    print(f"\n[pareto] repair csv 已生成到 {args.out_dir}/repair_*.csv，训练后用 collect_results 聚合 NDCG")


if __name__ == "__main__":
    main()
