#!/bin/bash

# 项目路径
cd /workspace/ljl/Pressure_forecasting/FKAN || exit 1

# 清除缓存
rm -rf train/__pycache__ model/__pycache__ utils/__pycache__

# 日志路径
LOG_FILE=train_$(date +%Y%m%d_%H%M%S).log

# 启动训练（后台+日志）

nohup python3 train_fkan.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "Training started with PID $TRAIN_PID"
wait "$TRAIN_PID"

# 查找最新的 Unet_i 目录（按编号升序）
SAVE_DIR=$(find results/ -maxdepth 1 -type d -name "fkan_*" | sort -V | tail -n 1)

# 复制 train_main.py 到最新的模型保存目录
if [ -d "$SAVE_DIR" ]; then
    cp train_fkan.py "$SAVE_DIR/train_main.py"
    echo "train_fkan.py 已复制到 $SAVE_DIR"
else
    cp train_fkan.py results/fkan/train_fkan.py
    echo "train_fkan.py 已复制到 results/fkan/train_fkan.py"
    echo "未找到有效的模型保存目录（results/fkan_i）"
fi