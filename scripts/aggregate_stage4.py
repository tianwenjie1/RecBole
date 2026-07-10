# -*- coding: utf-8 -*-
# Stage 4A 汇总：读所有 train log + calib json，产 5 个 CSV + stage4_summary.md + pass/fail 判定。
import os
import re
import glob
import json
from collections import defaultdict
import pandas as pd

STAGE4_DIR = "logs/stage4"
METHODS = ["noisy", "loss_reweight_rc", "pad_like_rc", "cfu_single_rc", "proposed"]
ALPHA = 0.01
METRICS = ["recall@20", "ndcg@20", "tailrecall@20", "tailndcg@20"]


def parse_test(path):
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "test result" in line:
                last = line
    if not last:
        return None
    pairs = re.findall(r"\('([\w@]+)',\s*np\.float64\(([\d.]+)\)\)", last)
    return {k: float(v) for k, v in pairs} if pairs else None


def load_calib(wd, method):
    p = os.path.join(wd, f"calib_{method}__a{ALPHA}.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        return d.get("deploy_validate", {})
    return {}


def main():
    rows = []
    for wd in sorted(glob.glob(os.path.join(STAGE4_DIR, "*_*_s*"))):
        base = os.path.basename(wd)
        m = re.match(r"(.+)_(early|middle|recent|uniform)_s(\d+)", base)
        if not m:
            continue
        ds, pos, seed = m.group(1), m.group(2), int(m.group(3))
        for method in METHODS:
            t = parse_test(os.path.join(wd, f"train_{method}.log"))
            if t is None:
                continue
            cal = load_calib(wd, method) if method != "noisy" else {}
            rows.append({
                "dataset": ds, "position": pos, "seed": seed, "method": method,
                "ndcg@20": t.get("ndcg@20"), "recall@20": t.get("recall@20"),
                "tailrecall@20": t.get("tailrecall@20"), "tailndcg@20": t.get("tailndcg@20"),
                "tail_FPR": cal.get("tail_FPR"), "clean_FPR": cal.get("clean_FPR"),
                "noise_precision": cal.get("noise_precision"), "noise_recall": cal.get("noise_recall"),
                "budget": cal.get("budget_actual"),
            })
    if not rows:
        print("无结果，检查 logs/stage4/ 是否有训练日志。"); return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(STAGE4_DIR, "per_seed_results.csv"), index=False)

    # overall: 平均 over seed, per (ds, pos, method)
    overall = df.groupby(["dataset", "position", "method"])[METRICS].mean().reset_index()
    overall.to_csv(os.path.join(STAGE4_DIR, "overall_results.csv"), index=False)

    # matched-risk: 平均 over pos, seed, per (ds, method) + 风险指标
    g = df.groupby(["dataset", "method"])
    mr = g[METRICS].mean().reset_index()
    risk = g[["tail_FPR", "clean_FPR", "noise_precision", "noise_recall", "budget"]].mean().reset_index()
    mr = mr.merge(risk, on=["dataset", "method"])
    mr.to_csv(os.path.join(STAGE4_DIR, "matched_risk_results.csv"), index=False)

    # group_risk: per method 平均风险
    gr = df.groupby("method")[["tail_FPR", "clean_FPR", "noise_precision", "noise_recall", "budget"]].mean().reset_index()
    gr.to_csv(os.path.join(STAGE4_DIR, "group_risk_results.csv"), index=False)

    # risk violation
    rv = df[df["method"] != "noisy"].copy()
    rv["violation"] = rv["tail_FPR"] > ALPHA + 1e-9
    rvs = rv.groupby("method")["violation"].agg(["mean", "sum", "count"]).reset_index()
    rvs.columns = ["method", "violation_rate", "n_violation", "n"]
    rvs.to_csv(os.path.join(STAGE4_DIR, "risk_violation_summary.csv"), index=False)

    # ===== 判定 =====
    lines = ["# Stage 4A 汇总判定\n"]
    lines.append(f"总结果数: {len(df)} (预期 180)\n")
    lines.append("\n## 各数据集 × 方法（matched-risk, 平均 over pos/seed）\n")
    lines.append(mr.pivot(index="dataset", columns="method", values="ndcg@20").round(4).to_string())
    lines.append("\n\n## TailNDCG@20\n")
    lines.append(mr.pivot(index="dataset", columns="method", values="tailndcg@20").round(4).to_string())
    lines.append("\n\n## 风险违反率（tail_FPR>0.01 的比例，应低）\n")
    lines.append(rvs.to_string(index=False))

    # 判定 1: proposed NDCG 最优或差≤0.5% in ≥2/3 ds
    ds_list = df["dataset"].unique()
    j1 = 0
    for ds in ds_list:
        sub = mr[mr["dataset"] == ds]
        best = sub["ndcg@20"].max()
        prop = sub[sub["method"] == "proposed"]["ndcg@20"].values
        if len(prop) and best - prop[0] <= 0.005:
            j1 += 1
    # 判定 2: proposed TailRecall & TailNDCG > loss_reweight in ≥2/3 ds
    j2 = 0
    for ds in ds_list:
        sub = mr[mr["dataset"] == ds]
        prop = sub[sub["method"] == "proposed"]
        lr = sub[sub["method"] == "loss_reweight_rc"]
        if len(prop) and len(lr):
            if prop["tailrecall@20"].values[0] > lr["tailrecall@20"].values[0] and \
               prop["tailndcg@20"].values[0] > lr["tailndcg@20"].values[0]:
                j2 += 1
    # 判定 3: proposed > loss_reweight in ≥3/4 positions (平均 over ds/seed)
    pos_list = df["position"].unique()
    j3 = 0
    for pos in pos_list:
        sub = df[df["position"] == pos].groupby("method")["ndcg@20"].mean()
        if "proposed" in sub and "loss_reweight_rc" in sub and sub["proposed"] > sub["loss_reweight_rc"]:
            j3 += 1
    # 判定 4: tail-FPR ≤ 0.01
    prop_violation = rvs[rvs["method"] == "proposed"]["violation_rate"].values
    j4 = prop_violation[0] if len(prop_violation) else 1.0

    lines.append(f"\n\n## 判定\n")
    lines.append(f"1. proposed NDCG 最优或差≤0.5% 的数据集数: {j1}/{len(ds_list)} (需≥2)\n")
    lines.append(f"2. proposed TailRecall+TailNDCG 均超 loss-reweight 的数据集数: {j2}/{len(ds_list)} (需≥2)\n")
    lines.append(f"3. proposed NDCG 超 loss-reweight 的位置数: {j3}/{len(pos_list)} (需≥3)\n")
    lines.append(f"4. proposed tail-FPR 违反率: {j4:.3f} (应接近0)\n")
    ok = j1 >= 2 and j2 >= 2 and j3 >= 3 and j4 < 0.1
    lines.append(f"\n**Stage 4A 结论: {'通过 ✅ → 方向具备二区投稿价值' if ok else '未通过 ❌ → 检查弱项/考虑降级'}**\n")

    with open("logs/stage4_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
