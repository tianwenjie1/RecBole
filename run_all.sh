#!/usr/bin/env bash
# ====================================================================
# 过夜流水线：Beauty + SASRec 噪声下降曲线 + CFU 区分能力 + CFU-weight 训练
# 日志带辨识度，存 logs/ 下；最后用 collect_results.py 汇总成 RESULTS.md
#
# 用法（服务器）:
#   cd RecBole && git pull && nohup bash run_all.sh > logs/run_all.log 2>&1 &
#   tail -f logs/run_all.log     # 想看进度时
# ====================================================================
set -u
mkdir -p logs saved

DS=amazon-beauty
MODEL=SASRec
EPOCHS=100
export TOKENIZERS_PARALLELISM=false

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*"; }

run_train() {  # $1=gpu $2=noise_type $3=noise_ratio $4=ckpt_dir $5=logtag
  local gpu=$1 nt=$2 nr=$3 ckpt=$4 tag=$5
  log "TRAIN start $tag  (gpu=$gpu noise=$nt ratio=$nr)"
  python run_recbole.py -m $MODEL -d $DS \
    --gpu_id=$gpu \
    --noise_type=$nt --noise_ratio=$nr --noise_seed=2024 \
    --checkpoint_dir=$ckpt --epochs=$EPOCHS \
    > "logs/train_${tag}.log" 2>&1
  log "TRAIN done  $tag  (exit=$?)"
}

# ---------- Phase 1: 5 个 setting，2 卡并行 ----------
log "===== Phase 1: 噪声下降曲线 (5 settings) ====="
run_train 2 none         0.0 saved/beauty_clean       beauty_sasrec_none_0       &
run_train 2 random       0.1 saved/beauty_rand10      beauty_sasrec_random_10    &
run_train 3 random       0.2 saved/beauty_rand20      beauty_sasrec_random_20    &
run_train 3 popularity   0.1 saved/beauty_pop10       beauty_sasrec_pop_10       &
run_train 2 popularity   0.2 saved/beauty_pop20       beauty_sasrec_pop_20       &
wait
log "===== Phase 1 done ====="

# ---------- Phase 2: CFU 打分（用 clean checkpoint 评 random-10 噪声行）----------
log "===== Phase 2: CFU scoring ====="
python scripts/cfu_scoring.py \
  --dataset $DS --model $MODEL --gpu_id 2 \
  --noise_type random --noise_ratio 0.1 --noise_seed 2024 \
  --checkpoint_dir saved/beauty_clean --sample_ratio 1.0 \
  --out logs/cfu_beauty_random_10.csv \
  > logs/cfu_beauty_random_10.log 2>&1
log "CFU scoring done (exit=$?)"

# popularity 噪声也打一份
python scripts/cfu_scoring.py \
  --dataset $DS --model $MODEL --gpu_id 2 \
  --noise_type popularity --noise_ratio 0.1 --noise_seed 2024 \
  --checkpoint_dir saved/beauty_clean --sample_ratio 1.0 \
  --out logs/cfu_beauty_pop_10.csv \
  > logs/cfu_beauty_pop_10.log 2>&1
log "CFU scoring pop done (exit=$?)"

# ---------- Phase 3: 生成 3 种权重 ----------
log "===== Phase 3: build weights ====="
python scripts/build_weights.py --cfu_csv logs/cfu_beauty_random_10.csv \
  --strategy cfu_only   --tau 0.2 --out logs/w_cfu_only.csv   > logs/w_cfu_only.log 2>&1
python scripts/build_weights.py --cfu_csv logs/cfu_beauty_random_10.csv \
  --strategy loss_reweight --tau 0.2 --out logs/w_loss_reweight.csv > logs/w_loss_reweight.log 2>&1
python scripts/build_weights.py --cfu_csv logs/cfu_beauty_random_10.csv \
  --strategy cfu_tail   --tau 0.2 --tail_min 0.5 --out logs/w_cfu_tail.csv > logs/w_cfu_tail.log 2>&1
log "build weights done"

# ---------- Phase 4: CFU-weighted 训练（Beauty random-10）----------
log "===== Phase 4: CFU-weighted training ====="
run_cfu_train() {  # $1=strategy $2=weight_file $3=logtag
  python run_recbole.py -m $MODEL -d $DS \
    --gpu_id=2 \
    --noise_type=random --noise_ratio=0.1 --noise_seed=2024 \
    --use_cfu_weight=True --cfu_weight_file=$2 \
    --checkpoint_dir=saved/beauty_$1 --epochs=$EPOCHS \
    > "logs/train_beauty_sasrec_$3.log" 2>&1
  log "CFU-TRAIN done $3 (exit=$?)"
}
run_cfu_train cfu_only    logs/w_cfu_only.csv            cfu_only
run_cfu_train loss_reweight logs/w_loss_reweight.csv loss_reweight
run_cfu_train cfu_tail    logs/w_cfu_tail.csv            cfu_tail
log "===== Phase 4 done ====="

# ---------- Phase 5: CFU-guided input repair 训练（救判断点 C）----------
log "===== Phase 5: input repair ====="
python scripts/build_repair.py --cfu_csv logs/cfu_beauty_random_10.csv \
  --selection percentile --low_pct 20 --action mask  --tail_protect 0 \
  --out logs/repair_cfu_mask.csv         > logs/repair_cfu_mask.log 2>&1
python scripts/build_repair.py --cfu_csv logs/cfu_beauty_random_10.csv \
  --selection percentile --low_pct 20 --action mask  --tail_protect 1 \
  --out logs/repair_cfu_mask_tail.csv    > logs/repair_cfu_mask_tail.log 2>&1
python scripts/build_repair.py --cfu_csv logs/cfu_beauty_random_10.csv \
  --selection percentile --low_pct 20 --action replace --tail_protect 1 \
  --out logs/repair_cfu_replace_tail.csv > logs/repair_cfu_replace_tail.log 2>&1

run_repair_train() {  # $1=strategy $2=repair_file $3=logtag
  python run_recbole.py -m $MODEL -d $DS \
    --gpu_id=2 \
    --noise_type=random --noise_ratio=0.1 --noise_seed=2024 --noise_position=last \
    --use_input_repair=True --repair_file=$2 \
    --checkpoint_dir=saved/beauty_$1 --epochs=$EPOCHS \
    > "logs/train_beauty_sasrec_$3.log" 2>&1
  log "REPAIR-TRAIN done $3 (exit=$?)"
}
run_repair_train cfu_mask         logs/repair_cfu_mask.csv         cfu_mask
run_repair_train cfu_mask_tail    logs/repair_cfu_mask_tail.csv    cfu_mask_tail
run_repair_train cfu_replace_tail logs/repair_cfu_replace_tail.csv cfu_replace_tail
log "===== Phase 5 done ====="

# ---------- 汇总 ----------
log "===== collect results ====="
python scripts/collect_results.py > logs/collect.log 2>&1
log "ALL DONE. see RESULTS.md + logs/"
echo "ALL DONE $(ts)" >> logs/run_all_marker.txt
