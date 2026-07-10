#!/usr/bin/env bash
# ====================================================================
# Stage 4A：三数据集确认矩阵（定位 A：风险受控选择性修复）
# 矩阵：3 数据集 × 4 噪声位置 × 3 seeds × 5 方法 = 180 训练
# 方法：noisy / loss_reweight_rc / pad_like_rc / cfu_single_rc / proposed(cf+弃权)
# GPU 2/3 并行，skip-if-done + timeout 1200。用法：
#   conda activate smore && nohup bash run_stage4a.sh > logs/stage4a.log 2>&1 &
# ====================================================================
set -u
mkdir -p logs saved
GPUS=(2 3)
ALPHA=0.01
EPOCHS=100
# dataset: category:recbole_name
DATASETS=("beauty:amazon-beauty" "toys:amazon-toys-games" "sports:amazon-sports")
POSITIONS=(early middle recent uniform)
SEEDS=(42 2026 3407)
METHODS=(noisy loss_reweight_rc pad_like_rc cfu_single_rc proposed)

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }

# ---- Phase 0: 数据 + clean checkpoint ----
for entry in "${DATASETS[@]}"; do
  cat="${entry%%:*}"; ds="${entry##*:}"
  if [ ! -f "dataset/${ds}/${ds}.inter" ]; then
    log "下载数据 $cat -> $ds"
    python scripts/get_amazon.py --category $cat || { log "数据下载失败 $cat"; continue; }
  fi
  if ! ls saved/${ds}_clean/SASRec-*.pth >/dev/null 2>&1; then
    log "训练 clean checkpoint $ds"
    python run_recbole.py -m SASRec -d $ds --gpu_id=${GPUS[0]} \
      --noise_type=none --noise_ratio=0.0 --noise_seed=42 \
      --checkpoint_dir=saved/${ds}_clean --epochs=$EPOCHS \
      > logs/stage4_clean_${ds}.log 2>&1
  fi
done

# ---- job 队列 ----
declare -a Q
i=0
enqueue() { Q+=("$1|$2|$3|$4|$5"); }   # gpu|ds|pos|seed|method

for entry in "${DATASETS[@]}"; do
  cat="${entry%%:*}"; ds="${entry##*:}"
  [ -f "dataset/${ds}/${ds}.inter" ] || { log "跳过 $ds（无数据）"; continue; }
  for pos in "${POSITIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      wd="logs/stage4/${ds}_${pos}_s${seed}"
      mkdir -p "$wd"
      cfu="$wd/cfu.csv"
      # 1. CFU 打分（单步 + dropout 不确定性；skip 若已存在）
      if [ ! -f "$cfu" ]; then
        log "CFU $ds $pos s$seed"
        python scripts/cfu_scoring.py --dataset $ds --model SASRec --gpu_id ${GPUS[0]} \
          --noise_type random --noise_ratio 0.1 --noise_seed $seed --noise_position $pos \
          --checkpoint_dir saved/${ds}_clean --horizons 1 --n_dropout 3 \
          --out "$cfu" > "$wd/cfu.log" 2>&1
      fi
      # 2. pareto_eval 生成 4 个 repair csv（含 proposed kappa=1.0，alpha=0.01）
      if [ ! -f "$wd/repair_proposed__a${ALPHA}.csv" ]; then
        python scripts/pareto_eval.py --cfu_csv "$cfu" --out_dir "$wd" \
          --alphas $ALPHA > "$wd/pareto.log" 2>&1
      fi
      # 3. 入队 5 个训练
      for m in "${METHODS[@]}"; do
        enqueue "${GPUS[$((i % 2))]}" "$ds" "$pos" "$seed" "$m"
        i=$((i+1))
      done
    done
  done
done

# ---- 跑队列：每批 2 个（一卡一个）----
log "===== 训练队列 ${#Q[@]} 个 ====="
n=0
for job in "${Q[@]}"; do
  IFS='|' read -r gpu ds pos seed m <<< "$job"
  wd="logs/stage4/${ds}_${pos}_s${seed}"
  logfile="$wd/train_${m}.log"
  # skip-if-done
  if grep -q "test result" "$logfile" 2>/dev/null; then continue; fi
  if [ "$m" = "noisy" ]; then
    cmd="python run_recbole.py -m SASRec -d $ds --gpu_id=$gpu \
      --noise_type=random --noise_ratio=0.1 --noise_seed=$seed --noise_position=$pos \
      --checkpoint_dir=saved/stage4_${ds}_${pos}_s${seed}_${m} --epochs=$EPOCHS"
  else
    rep="$wd/repair_${m}__a${ALPHA}.csv"
    cmd="python run_recbole.py -m SASRec -d $ds --gpu_id=$gpu \
      --noise_type=random --noise_ratio=0.1 --noise_seed=$seed --noise_position=$pos \
      --use_input_repair=True --repair_file=$rep \
      --checkpoint_dir=saved/stage4_${ds}_${pos}_s${seed}_${m} --epochs=$EPOCHS"
  fi
  timeout 1200 $cmd > "$logfile" 2>&1 &
  log "START ${ds} ${pos} s${seed} ${m} (gpu=$gpu)"
  n=$((n+1))
  if [ $((n % 2)) -eq 0 ]; then wait; fi
done
wait
log "===== 训练完成，汇总 ====="
python scripts/aggregate_stage4.py > logs/stage4_aggregate.log 2>&1
log "ALL DONE. see logs/stage4_summary.md"
echo "STAGE4A DONE $(ts)" >> logs/stage4a_marker.txt
