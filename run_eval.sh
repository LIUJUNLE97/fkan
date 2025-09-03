#!/bin/bash

# 项目路径
cd /workspace/ljl/Pressure_forecasting/FKAN || exit 1

# 清除缓存
# rm -rf eval/__pycache__ model/__pycache__ utils/__pycache__

# 日志路径
LOG_FILE=eval_$(date +%Y%m%d_%H%M%S).log

nohup python3 eval_fkan.py > "$LOG_FILE" 2>&1 &
eval_PID=$!
echo "Evaluation started with PID $eval_PID"
wait "$eval_PID"

# 查找最新的 Unet_i 目录（按编号升序）
SAVE_DIR=$(find results/ -maxdepth 1 -type d -name "fkan_*" | sort -V | tail -n 1)

# 复制 eval_main.py 到最新的模型保存目录
if [ -d "$SAVE_DIR" ]; then
    cp eval_fkan.py "$SAVE_DIR/eval_main.py"
    echo "eval_fkan.py 已复制到 $SAVE_DIR"
else
    cp eval_fkan.py results/fkan/eval_fkan.py
    echo "eval_fkan.py 已复制到 results/fkan/eval_fkan.py"
    echo "未找到有效的模型保存目录（results/fkan_i）"
fi