#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

MODEL_PATH=${1:-"DAMO-NLP-SG/VideoLLaMA3-7B"}
BENCHMARKS=${2:-"mvbench,videomme,tempcompass,charades_sta,activitynet_tg,hc_stvg_v1,hc_stvg_v2,vidstg,v-star"}

ARG_WORLD_SIZE=${3:-1}
ARG_NPROC_PER_NODE=${4:-8}

ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16667
ARG_RANK=${6:-0}

if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi


echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MODEL_PATH: $MODEL_PATH"
echo "BENCHMARKS: $BENCHMARKS"


SAVE_DIR=evaluation_results
DATA_ROOT=dataroot
declare -A DATA_ROOTS

# mcqa
DATA_ROOTS["mvbench"]="$DATA_ROOT/MVBench"
DATA_ROOTS["videomme"]="$DATA_ROOT/Video-MME"
DATA_ROOTS["tempcompass"]="$DATA_ROOT/TempCompass"
DATA_ROOTS["charades_sta"]="$DATA_ROOT/charades-sta-test"
DATA_ROOTS["hc_stvg_v1"]="$DATA_ROOT/HC-STVG/data"
DATA_ROOTS["hc_stvg_v2"]="$DATA_ROOT/HC-STVG/data"
DATA_ROOTS["vidstg"]="$DATA_ROOT/VidSTG"
DATA_ROOTS["v-star"]="$DATA_ROOT/V-STaR"
DATA_ROOTS["nextgqa"]="$DATA_ROOT/NEXTGQA"

IFS=',' read -ra BENCHMARK_LIST <<< "$BENCHMARKS"
for BENCHMARK in "${BENCHMARK_LIST[@]}"; do
    DATA_ROOT=${DATA_ROOTS[$BENCHMARK]}
    if [ -z "$DATA_ROOT" ]; then
        echo "Error: Data root for benchmark '$BENCHMARK' not defined."
        continue
    fi
    torchrun --nnodes $WORLD_SIZE \
        --nproc_per_node $NPROC_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --node_rank $RANK \
        evaluation/evaluate.py \
        --model_path ${MODEL_PATH} \
        --benchmark ${BENCHMARK} \
        --data_root ${DATA_ROOT} \
        --save_path "${SAVE_DIR}/${MODEL_PATH##*/}/${BENCHMARK}.json" \
        --fps 1 \
        --max_frames 180 \
        --max_visual_tokens 16384
done