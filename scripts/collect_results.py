# -*- coding: utf-8 -*-
# 解析日志生成 RESULTS.md。
# 支持 sweep 日志（sweep_train_<ds>_random_10_<pos>_<sel>_seed<seed>.log）+ repair_*.json 诊断。
# 同时解析 valid 和 test，按 valid NDCG@20 选最优，报告对应 test。
import os
import re
import glob
import json
import ast
from datetime import datetime

LOG_DIR = "logs"
OUT = "RESULTS.md"

METRIC_KEYS = ["recall@20", "ndcg@20", "tailrecall@20", "tailndcg@20"]


def parse_result_line(path, keyword):
    """从 log 抓最后一条 keyword 行（'valid result'/'test result'），解析 OrderedDict。
    RecBole 的 valid result: 和指标可能分两行，所以也看 keyword 行的下一行。"""
    if not os.path.exists(path):
        return None
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()
    idxs = [i for i, ln in enumerate(lines) if keyword in ln]
    if not idxs:
        return None
    idx = idxs[-1]
    # 试 keyword 行本身 + 下一行（指标可能在下一行）
    for cand in [lines[idx], lines[idx + 1] if idx + 1 < len(lines) else ""]:
        pairs = re.findall(r"\('([\w@]+)',\s*np\.float64\(([\d.]+)\)\)", cand)
        if not pairs:
            # 兼容 'recall@20 : 0.2246' 这种空格分隔格式
            sp = re.findall(r"([\w@]+)\s*:\s*([\d.]+)", cand)
            if sp and any(k in ["recall@20", "ndcg@20"] for k, _ in sp):
                return {k: float(v) for k, v in sp}
        else:
            return {k: float(v) for k, v in pairs}
    return None


def fmt(d, key):
    if not d or key not in d:
        return "-"
    return f"{d[key]:.4f}"


def load_repair_json(pos, sel):
    # repair_<pos>_<sel>.json
    p = os.path.join(LOG_DIR, f"repair_{pos}_{sel}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def main():
    lines = [f"# 实验结果汇总（自动生成 {datetime.now():%F %T}）\n"]
    lines.append("数据集: Amazon Beauty (2014 5-core) | 模型: SASRec | noise=random 10%\n")

    # ===== sweep 表（读 meta json）=====
    metas = sorted(glob.glob(os.path.join(LOG_DIR, "meta__*.json")))
    if metas:
        lines.append("\n## Sweep：噪声位置 × 选择策略（seed 42）\n")
        lines.append("按 valid NDCG@20 选最优；test 为该阈值对应值。")
        lines.append("\n| position | selection | NDCG@20(valid) | NDCG@20(test) | Recall@20 | TailRecall@20 | TailNDCG@20 | noise_prec | noise_recall | tail_FRR |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        rows = []
        for mf in metas:
            with open(mf) as f:
                meta = json.load(f)
            pos, sel, seed = meta["pos"], meta["sel"], meta.get("seed", "")
            log = meta["log"]
            valid = parse_result_line(log, "valid result")
            test = parse_result_line(log, "test result")
            diag = {}
            rj = meta.get("repair_json")
            if rj and os.path.exists(rj):
                with open(rj) as f:
                    diag = json.load(f)
            rows.append({
                "pos": pos, "sel": sel, "seed": seed,
                "valid": valid, "test": test, "diag": diag,
            })
        # 排序：position, 然后 valid ndcg 降序
        rows.sort(key=lambda r: (r["pos"], -((r["valid"] or {}).get("ndcg@20", -1))))
        for r in rows:
            v, t, d = r["valid"], r["test"], r["diag"]
            def df(key):
                val = d.get(key)
                return f"{val:.4f}" if isinstance(val, (int, float)) else "-"
            lines.append(
                f"| {r['pos']} | {r['sel']} | {fmt(v,'ndcg@20')} | {fmt(t,'ndcg@20')} "
                f"| {fmt(t,'recall@20')} | {fmt(t,'tailrecall@20')} | {fmt(t,'tailndcg@20')} "
                f"| {df('noise_precision')} | {df('noise_recall')} | {df('tail_false_repair_rate')} |"
            )
        # 每个 position 的 best-by-valid
        lines.append("\n### 各 position 下 valid 最优 selection\n")
        lines.append("| position | best selection | valid NDCG@20 | test NDCG@20 | test TailNDCG@20 |")
        lines.append("|---|---|---|---|---|")
        seen = {}
        for r in rows:
            if r["pos"] not in seen:
                seen[r["pos"]] = r
        for pos, r in seen.items():
            lines.append(f"| {pos} | {r['sel']} | {fmt(r['valid'],'ndcg@20')} | {fmt(r['test'],'ndcg@20')} | {fmt(r['test'],'tailndcg@20')} |")

    # ===== 旧 run_all 表（若存在）=====
    lines.append("\n## 旧 run_all 结果（若 sweep 未覆盖）\n")
    old_tags = {
        "none_0": "Clean", "random_10": "Noisy", "random_20": "Random20",
        "pop_10": "PopNoise10", "pop_20": "PopNoise20",
        "loss_reweight": "loss-reweight", "cfu_mask": "CFU-mask",
        "cfu_mask_tail": "CFU-mask+tail", "cfu_replace_tail": "CFU-replace+tail",
    }
    lines.append("\n| Setting | NDCG@20 | Recall@20 | TailNDCG@20 |")
    lines.append("|---|---|---|---|")
    for tag, desc in old_tags.items():
        t = parse_result_line(os.path.join(LOG_DIR, f"train_beauty_sasrec_{tag}.log"), "test result")
        if t:
            lines.append(f"| {desc} | {fmt(t,'ndcg@20')} | {fmt(t,'recall@20')} | {fmt(t,'tailndcg@20')} |")

    # ===== CFU 分离（命门 B）=====
    lines.append("\n## CFU 分离（命门 B）\n")
    for cfu_log in sorted(glob.glob(os.path.join(LOG_DIR, "cfu_*_random_10_*.log"))):
        base = os.path.basename(cfu_log).replace(".log", "")
        with open(cfu_log, "r", errors="ignore") as f:
            txt = f.read()
        idx = txt.find("===== CFU separation")
        lines.append(f"\n### {base}")
        lines.append("```\n" + (txt[idx:].strip() if idx >= 0 else "(无统计)") + "\n```")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
