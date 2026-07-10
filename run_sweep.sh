#!/usr/bin/env bash
# ====================================================================
# Part 1 sweep：Beauty random-10，阈值 × 噪声位置 全扫（seed 42 先跑）
# 日志名含 dataset/noise_type/ratio/position/selection/threshold/seed
# 用 GPU 3、4 并行。用法：
#   conda activate smore
#   nohup bash run_sweep.sh > logs/run_sweep.log 2>&1 &
# ====================================================================
set -u
mkdir -p logs saved

DS=amazon-beauty
MODEL=SASRec
EPOCHS=100
SEED=42
RATIO=0.1
GPUS=(3 4)          # ← 改这里换卡
POSITIONS=(last recent middle early uniform)
# selection 列表："kind|arg"，kind=pct 用 low_pct，kind=raw 用 raw_negative
SELECTIONS=("pct|5" "pct|7.5" "pct|10" "pct|12.5" "pct|15" "pct|20" "raw|0")

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }

# 确保 clean checkpoint 存在
if ! ls saved/beauty_clean/${MODEL}-*.pth >/dev/null 2>&1; then
  log "clean checkpoint 缺失，先训练 clean..."
  python run_recbole.py -m $MODEL -d $DS --gpu_id=${GPUS[0]} \
    --noise_type=none --noise_ratio=0.0 --noise_seed=$SEED \
    --checkpoint_dir=saved/beauty_clean --epochs=$EPOCHS \
    > logs/sweep_clean.log 2>&1
fi

# job 队列：每 2 个（分到 2 卡）一批，wait
declare -a QUEUE
enqueue() { QUEUE+=("$1|$2|$3|$4"); }   # gpu|pos|selkind|selarg

for pos in "${POSITIONS[@]}"; do
  # 1. cfu 打分（带 position 标签）
  log "CFU scoring position=$pos"
  python scripts/cfu_scoring.py --dataset $DS --model $MODEL --gpu_id ${GPUS[0]} \
    --noise_type random --noise_ratio $RATIO --noise_seed $SEED --noise_position $pos \
    --checkpoint_dir saved/beauty_clean --sample_ratio 1.0 \
    --out logs/cfu_${DS}_random_10_${pos}.csv \
    > logs/cfu_${DS}_random_10_${pos}.log 2>&1
  CFU=logs/cfu_${DS}_random_10_${pos}.csv

  # 2. baseline: noisy（无 repair）
  enqueue "${GPUS[0]}" "$pos" "noisy" "-"
  # baseline: loss-reweight
  python scripts/build_weights.py --cfu_csv $CFU --strategy loss_reweight --tau 0.2 \
    --out logs/w_loss_reweight_${pos}.csv > logs/w_loss_reweight_${pos}.log 2>&1
  enqueue "${GPUS[1]}" "$pos" "loss_reweight" "-"

  # 3. 各 selection 的 repair
  g=0
  for sel in "${SELECTIONS[@]}"; do
    kind="${sel%%|*}"; arg="${sel##*|}"
    if [ "$kind" = "pct" ]; then
      selname="pct${arg}"; selargs="--selection percentile --low_pct $arg"
    else
      selname="raw_negative"; selargs="--selection raw_negative"
    fi
    repcsv=logs/repair_${pos}_${selname}.csv
    python scripts/build_repair.py --cfu_csv $CFU $selargs \
      --action replace --tail_protect 1 --out $repcsv \
      > logs/repair_${pos}_${selname}.log 2>&1
    enqueue "${GPUS[$((g % 2))]}" "$pos" "$selname" "$repcsv"
    g=$((g+1))
  done
done

# 跑队列：每批 2 个（一卡一个），wait
log "===== 训练队列 ${#QUEUE[@]} 个 ====="
i=0
for job in "${QUEUE[@]}"; do
  IFS='|' read -r gpu pos selname repcsv <<< "$job"
  if [ "$selname" = "noisy" ]; then
    cmd="python run_recbole.py -m $MODEL -d $DS --gpu_id=$gpu \
      --noise_type=random --noise_ratio=$RATIO --noise_seed=$SEED --noise_position=$pos \
      --checkpoint_dir=saved/beauty_${pos}_noisy --epochs=$EPOCHS"
  elif [ "$selname" = "loss_reweight" ]; then
    cmd="python run_recbole.py -m $MODEL -d $DS --gpu_id=$gpu \
      --noise_type=random --noise_ratio=$RATIO --noise_seed=$SEED --noise_position=$pos \
      --use_cfu_weight=True --cfu_weight_file=logs/w_loss_reweight_${pos}.csv \
      --checkpoint_dir=saved/beauty_${pos}_loss_reweight --epochs=$EPOCHS"
  else
    cmd="python run_recbole.py -m $MODEL -d $DS --gpu_id=$gpu \
      --noise_type=random --noise_ratio=$RATIO --noise_seed=$SEED --noise_position=$pos \
      --use_input_repair=True --repair_file=$repcsv \
      --checkpoint_dir=saved/beauty_${pos}_${selname} --epochs=$EPOCHS"
  fi
  tag="${pos}__${selname}__seed${SEED}"
  logfile="logs/sweep_train__${tag}.log"
  # 写 meta json 方便 collect_results 解析（避免文件名歧义）
  python -c "import json; json.dump({'pos':'$pos','sel':'$selname','seed':'$SEED','log':'$logfile','repair_json':'logs/repair_${pos}_${selname}.json'}, open('logs/meta__${tag}.json','w'))"
  $cmd > "$logfile" 2>&1 &
  log "START $tag (gpu=$gpu)"
  i=$((i+1))
  if [ $((i % 2)) -eq 0 ]; then wait; log "batch done"; fi
done
wait
log "===== 训练完成 ====="

python scripts/collect_results.py > logs/collect.log 2>&1
log "ALL DONE. see RESULTS.md"
echo "SWEEP DONE $(ts)" >> logs/run_sweep_marker.txt
