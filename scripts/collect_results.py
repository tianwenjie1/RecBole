# -*- coding: utf-8 -*-
# 解析 logs/train_*.log 里的 'test result' 行 + cfu 打分统计，汇总成 RESULTS.md
import os
import re
import glob
import ast
from datetime import datetime

LOG_DIR = "logs"
OUT = "RESULTS.md"

# train_beauty_sasrec_<tag>.log  ->  tag 描述
TAG_DESC = {
    "none_0": ("Clean SASRec", "none", 0.0),
    "random_10": ("Random 10%", "random", 0.1),
    "random_20": ("Random 20%", "random", 0.2),
    "pop_10": ("PopNoise 10%", "popularity", 0.1),
    "pop_20": ("PopNoise 20%", "popularity", 0.2),
    "cfu_only": ("Noisy + CFU-only", "random", 0.1),
    "loss_reweight": ("Noisy + loss-reweight", "random", 0.1),
    "cfu_tail": ("Noisy + CFU+tail", "random", 0.1),
    "cfu_mask": ("Noisy + CFU-mask repair", "random", 0.1),
    "cfu_mask_tail": ("Noisy + CFU-mask+tail", "random", 0.1),
    "cfu_replace_tail": ("Noisy + CFU-replace+tail", "random", 0.1),
}


def parse_test_result(path):
    """从 log 里抓最后一行 'test result: OrderedDict([...])' 解析成 dict。"""
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "test result" in line:
                last = line
    if last is None:
        return None
    # 形如: ('recall@20', np.float64(0.2246))
    pairs = re.findall(r"\('([\w@]+)',\s*np\.float64\(([\d.]+)\)\)", last)
    if not pairs:
        # 兼容旧版纯 dict 格式 {'recall@10': 0.x, ...}
        m = re.search(r"test result.*?(\{.*\})", last)
        if m:
            try:
                return ast.literal_eval(m.group(1))
            except Exception:
                return None
        return None
    return {k: float(v) for k, v in pairs}


def fmt(d, key):
    if d is None or key not in d:
        return "-"
    return f"{d[key]:.4f}"


def main():
    lines = []
    lines.append(f"# 实验结果汇总（自动生成 {datetime.now():%F %T}）\n")
    lines.append("数据集: Amazon Beauty (5-core) | 模型: SASRec | seed=2024\n")

    # ---- 表1: 噪声下降曲线 ----
    lines.append("\n## 表1 噪声下降曲线 (判断点 A)\n")
    lines.append("| Setting | Recall@20 | NDCG@20 | TailRecall@20 | TailNDCG@20 |")
    lines.append("|---|---|---|---|---|")
    order = ["none_0", "random_10", "random_20", "pop_10", "pop_20"]
    for tag in order:
        desc = TAG_DESC[tag][0]
        r = parse_test_result(os.path.join(LOG_DIR, f"train_beauty_sasrec_{tag}.log"))
        lines.append(f"| {desc} | {fmt(r,'recall@20')} | {fmt(r,'ndcg@20')} "
                     f"| {fmt(r,'tailrecall@20')} | {fmt(r,'tailndcg@20')} |")
    lines.append("\n> 判定：噪声下 NDCG@20 / TailNDCG@20 应明显低于 Clean。")

    # ---- 表3: CFU-weight 对比 ----
    lines.append("\n## 表3 CFU-weight 性能保持 (判断点 C, Beauty random-10)\n")
    lines.append("| Setting | Recall@20 | NDCG@20 | TailRecall@20 | TailNDCG@20 |")
    lines.append("|---|---|---|---|---|")
    for tag in ["none_0", "random_10", "loss_reweight", "cfu_only", "cfu_tail",
                "cfu_mask", "cfu_mask_tail", "cfu_replace_tail"]:
        desc = TAG_DESC[tag][0]
        r = parse_test_result(os.path.join(LOG_DIR, f"train_beauty_sasrec_{tag}.log"))
        lines.append(f"| {desc} | {fmt(r,'recall@20')} | {fmt(r,'ndcg@20')} "
                     f"| {fmt(r,'tailrecall@20')} | {fmt(r,'tailndcg@20')} |")
    lines.append("\n> 判定：CFU-only/CFU+tail 应优于 Noisy 与 loss-reweight。")

    # ---- 表2: CFU 区分能力 ----
    lines.append("\n## 表2 CFU 区分能力 (判断点 B 命门)\n")
    for tag in ["random_10", "pop_10"]:
        cfu_log = os.path.join(LOG_DIR, f"cfu_beauty_{tag}.log")
        lines.append(f"\n### {tag}")
        if os.path.exists(cfu_log):
            with open(cfu_log, "r", errors="ignore") as f:
                txt = f.read()
            # 抓 ===== CFU separation ===== 到结尾
            idx = txt.find("===== CFU separation")
            if idx >= 0:
                block = txt[idx:].strip()
                lines.append("```\n" + block + "\n```")
            else:
                lines.append("(CFU 统计未找到，检查 cfu log)")
        else:
            lines.append(f"(缺失 {cfu_log})")

    # ---- TMR: 从 build_weights 日志抓 ----
    lines.append("\n## 表4 TMR (判断点 D)\n")
    for tag in ["loss_reweight", "cfu_only", "cfu_tail"]:
        wlog = os.path.join(LOG_DIR, f"w_{tag}.log")
        lines.append(f"\n### {tag}")
        if os.path.exists(wlog):
            with open(wlog, "r", errors="ignore") as f:
                txt = f.read()
            for ln in txt.splitlines():
                if "TMR" in ln or "noise down" in ln or "weight stats" in ln:
                    lines.append(ln)
        else:
            lines.append(f"(缺失 {wlog})")

    lines.append("\n## 表5 input repair 召回/误伤（build_repair 日志）\n")
    for tag in ["cfu_mask", "cfu_mask_tail", "cfu_replace_tail"]:
        rlog = os.path.join(LOG_DIR, f"repair_{tag}.log")
        lines.append(f"\n### {tag}")
        if os.path.exists(rlog):
            with open(rlog, "r", errors="ignore") as f:
                for ln in f:
                    if "repaired" in ln or "recall" in ln or "误伤" in ln or "Counter" in ln or "low_thr" in ln:
                        lines.append(ln.rstrip())
        else:
            lines.append(f"(缺失 {rlog})")

    lines.append("\n## CFU 分布图\n")
    for p in sorted(glob.glob("logs/cfu_*.png")):
        lines.append(f"- {p}")

    lines.append("\n---\n## 原始日志\n")
    for p in sorted(glob.glob("logs/*.log")):
        lines.append(f"- {p}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
