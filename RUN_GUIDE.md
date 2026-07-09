# RUN_GUIDE.md — 服务器自检 + 过夜运行

## 0. 拉代码
```bash
cd ~/RecBole            # 或你放 RecBole 的目录
git pull
```

## 1. 环境自检（先跑这个，确认能跑起来再睡觉）
```bash
conda activate smore
python -c "import torch, recbole, pandas; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"
# 若 ImportError: No module named 'recbole'  ->  pip install -e .
# 若缺 torch/pandas/matplotlib/scikit-learn  ->  pip install <pkg>
nvidia-smi            # 确认 GPU 2、3 空闲
```

冒烟测试（clean，1 epoch，确认数据自动下载 + tail 指标 + 噪声开关都通）：
```bash
CUDA_VISIBLE_DEVICES=2 python run_recbole.py -m SASRec -d amazon-beauty \
  --noise_type=none --noise_ratio=0.0 --epochs=1 --checkpoint_dir=saved/smoke
```
看到 `test result: {... 'tailrecall@20': ... 'tailndcg@20': ...}` 就算通。

再测噪声注入 + CFU 权重通路（1 epoch）：
```bash
CUDA_VISIBLE_DEVICES=2 python run_recbole.py -m SASRec -d amazon-beauty \
  --noise_type=random --noise_ratio=0.1 --epochs=1 --checkpoint_dir=saved/smoke2
ls logs/noise_amazon-beauty_random_10_seed2024.csv   # 应有噪声日志
```

## 2. 过夜流水线（自检通过后，后台跑）
```bash
nohup bash run_all.sh > logs/run_all.log 2>&1 &
echo $! > logs/run_all.pid
tail -f logs/run_all.log     # 想看进度时；Ctrl-C 不会停止后台任务
```

流水线阶段（全部日志带辨识度存 `logs/`）：
1. **Phase 1**：5 个 setting 训练（clean / random 10,20 / pop 10,20），GPU 2、3 并行。
   日志：`logs/train_beauty_sasrec_<tag>.log`
2. **Phase 2**：CFU 打分（clean checkpoint 评 random-10 / pop-10 噪声行）。
   日志：`logs/cfu_beauty_*.log`，CSV：`logs/cfu_beauty_*.csv`，图：`logs/cfu_beauty_*.png`
3. **Phase 3**：生成 3 种权重。日志：`logs/w_*.log`
4. **Phase 4**：CFU-weighted 训练（cfu_only / loss_reweight / cfu_tail）。
5. **汇总**：`python scripts/collect_results.py` 生成 `RESULTS.md`。

## 3. 明早看结果
- `RESULTS.md`：表1 噪声曲线 / 表2 CFU 区分 / 表3 CFU-weight 对比 / 表4 TMR。
- `PROGRESS.md`：工作日志。
- 出问题看 `logs/run_all.log` 和对应 `logs/train_*.log`。

## 4. 判断点
- **A**：表1 噪声下 NDCG@20/TailNDCG@20 应明显低于 Clean。
- **B（命门）**：表2 CFU_delete 在 clean_real vs injected_noise 均值有差异，AUC>0.6。
- **C**：表3 CFU-only/CFU+tail 优于 Noisy + loss-reweight。
- **D**：表4 CFU+tail 的 TMR < CFU-only。

任一严重不达标 → 停下商量是否回退方向 1。

## 备注
- 停止过夜任务：`kill $(cat logs/run_all.pid)`；并行子任务用 `pkill -f run_recbole`。
- Beauty 小，单卡 12GB 足够；2/3 卡并行训练阶段，CFU/权重阶段只用 GPU2。
