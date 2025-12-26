#!/bin/bash

# ============================================================
# PRD-Net 主实验训练脚本
# 默认训练3个随机种子
#
# 用法:
#   ./train.sh [数据集] [--gpu GPU_ID] [--seeds 'SEED1 SEED2...'] [其他参数覆盖...]
#
# 示例:
#   ./train.sh mosi --gpu 0
#   ./train.sh sims --gpu 1 --seeds '1111'
#   ./train.sh mosei --gpu 0 --seeds '1111 1112 1113'
#   ./train.sh mosi --gpu 0 --model.prm.temperature=0.1
# ============================================================

# 激活Pytorch环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Pytorch

# 清理函数
cleanup() {
    echo ""
    echo "检测到 Ctrl+C，正在清理..."
    pkill -9 -u $(whoami) -f "python.*train.py" 2>/dev/null
    sleep 1
    echo "清理完成！"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ============================================================
# 默认参数
# ============================================================
GPU_ID=0
SEEDS=(1111 1112 1113)  # 默认三个随机种子
DATASETS=()
RUN_EVAL=false  # 是否在训练后运行评估

# ============================================================
# 数据集配置映射
# ============================================================
declare -A DATASET_CONFIGS
DATASET_CONFIGS["sims"]="configs/train_sims.yaml"
DATASET_CONFIGS["mosi"]="configs/train_mosi.yaml"
DATASET_CONFIGS["mosei"]="configs/train_mosei.yaml"

# ============================================================
# 解析命令行参数
# ============================================================
OVERRIDE_PARAMS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --seeds)
            IFS=' ' read -r -a SEEDS <<< "$2"
            shift 2
            ;;
        --eval)
            RUN_EVAL=true
            shift
            ;;
        --*)
            # 收集所有 --xxx=yyy 或 --xxx.yyy=zzz 格式的参数
            OVERRIDE_PARAMS+=("$1")
            shift
            ;;
        *)
            # 收集数据集参数 (支持空格分隔或逗号分隔)
            if [[ "$1" == *","* ]]; then
                IFS=',' read -ra ADDS <<< "$1"
                DATASETS+=("${ADDS[@]}")
            else
                DATASETS+=("$1")
            fi
            shift
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 如果没有指定数据集，使用默认
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("sims")
fi

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "开始处理数据集: ${DATASET}"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    # 获取配置文件
    CONFIG_FILE=${DATASET_CONFIGS[$DATASET]}

if [ -z "$CONFIG_FILE" ]; then
    echo "错误: 未知数据集 '$DATASET'"
    echo "支持的数据集: sims, mosi, mosei"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 评估指标配置
if [ "$DATASET" == "sims" ]; then
    KEY_EVALS=("Mult_acc_2" "Mult_acc_3" "Mult_acc_5" "MAE")
else
    KEY_EVALS=("Has0_acc_2" "Non0_acc_2" "Mult_acc_5" "Mult_acc_7" "MAE")
fi

# ============================================================
# 显示配置信息
# ============================================================
echo "============================================================"
echo "PRD-Net 主实验训练"
echo "============================================================"
echo "数据集: ${DATASET}"
echo "配置文件: ${CONFIG_FILE}"
echo "GPU: ${GPU_ID}"
echo "随机种子: ${SEEDS[@]}"
echo "训练后评估: ${RUN_EVAL}"
if [ ${#OVERRIDE_PARAMS[@]} -gt 0 ]; then
    echo "参数覆盖:"
    for param in "${OVERRIDE_PARAMS[@]}"; do
        echo "  ${param}"
    done
fi
echo "============================================================"
echo ""

# ============================================================
# 训练阶段：跑多个种子
# ============================================================
LOG_DIR="./log/main_experiment/${DATASET}"
mkdir -p ${LOG_DIR}

for seed in "${SEEDS[@]}"; do
    echo ""
    echo ">>> 训练 Seed=${seed}"

    TRAIN_CMD="python -u train.py --config_file ${CONFIG_FILE} --seed ${seed}"

    # 添加覆盖参数
    for param in "${OVERRIDE_PARAMS[@]}"; do
        TRAIN_CMD="${TRAIN_CMD} ${param}"
    done

    echo ">>> 命令: ${TRAIN_CMD}"

    eval ${TRAIN_CMD} 2>&1 | tee ${LOG_DIR}/train_seed_${seed}.log

    TRAIN_EXIT_CODE=$?
    if [ $TRAIN_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "训练失败！退出码: ${TRAIN_EXIT_CODE}"
        exit $TRAIN_EXIT_CODE
    fi

    echo ">>> 完成训练 Seed=${seed}"
    sleep 2
done

echo ""
echo "============================================================"
echo "所有种子训练完成！"
echo "============================================================"

# 注意：评估部分已分离，请使用 robust_evaluation.py 进行评估

done

echo ""
echo "============================================================"
echo "所有数据集处理完成！"
echo "============================================================"

