# PROGRESS.md — 鲁棒序列推荐去噪实验工作日志

方向：Counterfactual Utility-Aware Tail-Preserving Denoising for Robust Sequential Recommendation
命门：CFU 能否区分「注入噪声」与「真实 tail hard interaction」。

---

## 2026-07-09 晚：搭流水线（本地编码，待服务器自检后过夜跑）

### 做了什么
- 探索 RecBole 全链路（SASRec / 数据流 / 评估 / 配置），确认落地方式。
- 关键简化：RecBole 滑动窗口让每行训练样本 = (历史序列, 下一目标)，故 **per-row 权重 = per-position 权重**，无需改 SASRec.forward。
- 噪声注入设计为「污染选中行的最后一个历史 item」，与 CFU 评估位置严格对齐。

### 新增/修改文件
| 文件 | 说明 |
|---|---|
| `recbole/properties/dataset/amazon-beauty.yaml` | Beauty 5-core + LS/TO/full + tail 指标 + 噪声开关 |
| `recbole/properties/dataset/amazon-toys-games.yaml` | Toys 配置（Hour 12 备用） |
| `recbole/properties/overall.yaml` | 加 noise_*/cfu_*/tail_ratio 默认值 |
| `recbole/data/noise_inject.py` | 三种噪声注入 + noise log csv |
| `recbole/utils/cfu_weight.py` | 把 CFU 权重挂到 inter_feat |
| `recbole/quick_start/quick_start.py` | split 后接噪声注入 + CFU 权重 |
| `recbole/evaluator/collector.py` | 收集每用户正样本 item id（tail 指标用） |
| `recbole/evaluator/metrics.py` | 新增 TailRecall / TailNDCG（自动注册） |
| `recbole/model/sequential_recommender/sasrec.py` | CE 分支 per-row 权重化 |
| `scripts/cfu_scoring.py` | CFU 离线打分 + 统计 + 分布图 |
| `scripts/build_weights.py` | cfu_only / loss_reweight / cfu_tail 三种权重 + TMR 预统计 |
| `scripts/collect_results.py` | 解析日志生成 RESULTS.md |
| `run_all.sh` | 过夜流水线（2/3 卡并行训练 + 串行 CFU/权重/CFU训练） |

### 链路核对（已确认无设备/字段丢失问题）
- Interaction.to(device) 无 selected_field → cfu_weight 随 batch 上 GPU。
- SASRec CE 无 neg sampling → cfu_weight 字段原样流到 calculate_loss。
- tail 指标需 full-sort（sequential 默认 mode=full），positive_i 为全局 item id。
- checkpoint 路径 `{checkpoint_dir}/{model}-{timestamp}.pth`，run_all.sh 给每实验独立 checkpoint_dir，CFU 脚本 glob 读取。

### 待服务器执行
1. 自检：见 `RUN_GUIDE.md`。
2. 过夜：`nohup bash run_all.sh > logs/run_all.log 2>&1 &`，明早看 `RESULTS.md`。

### 下一步（明早根据结果决定）
- 判断点 A（噪声下降）、B（CFU 区分，命门）、C（CFU 优于 loss-reweight）、D（tail-preserving 降 TMR）。
- 任一止损命中 → 停下商量是否回退方向 1。
- 全过 → 补 Toys、GRU4Rec。
