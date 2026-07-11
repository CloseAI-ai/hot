#!/bin/bash
# HOT 训练状态检查脚本

echo "=========================================="
echo "  HOT 训练状态检查"
echo "=========================================="
echo ""

# 检查进程
echo "【进程状态】"
TRAIN_PROC=$(ps aux | grep "scripts/train.py" | grep -v grep | head -1)
MONITOR_PROC=$(ps aux | grep "train_monitor.py" | grep -v grep | head -1)
WEB_PROC=$(ps aux | grep "monitor.py.*8081" | grep -v grep | head -1)

if [ -n "$TRAIN_PROC" ]; then
    PID=$(echo "$TRAIN_PROC" | awk '{print $2}')
    CPU=$(echo "$TRAIN_PROC" | awk '{print $10}')
    MEM=$(echo "$TRAIN_PROC" | awk '{print $11}')
    echo "  训练进程: ✓ 运行中 (PID: $PID, CPU: $CPU%, MEM: $MEM)"
else
    echo "  训练进程: ✗ 未运行"
fi

if [ -n "$MONITOR_PROC" ]; then
    PID=$(echo "$MONITOR_PROC" | awk '{print $2}')
    echo "  监控器:   ✓ 运行中 (PID: $PID)"
else
    echo "  监控器:   ✗ 未运行"
fi

if [ -n "$WEB_PROC" ]; then
    PID=$(echo "$WEB_PROC" | awk '{print $2}')
    echo "  Web面板:  ✓ 运行中 (PID: $PID)"
else
    echo "  Web面板:  ✗ 未运行"
fi

echo ""

# GPU 状态
echo "【GPU 状态】"
nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader 2>/dev/null | while IFS=',' read -r temp mem_used mem_total util power; do
    echo "  温度: ${temp}°C"
    echo "  显存: ${mem_used} / ${mem_total}"
    echo "  利用率: ${util}"
    echo "  功耗: ${power}"
done

echo ""

# 训练进度
echo "【训练进度】"
if [ -f "train_final.log" ]; then
    LAST_STEP=$(grep -oP 'step \K\d+' train_final.log | tail -1)
    LAST_LOSS=$(grep -oP 'loss=\K[\d.]+' train_final.log | tail -1)
    if [ -n "$LAST_STEP" ]; then
        echo "  当前步骤: $LAST_STEP"
        echo "  当前损失: $LAST_LOSS"
        PROGRESS=$(echo "scale=1; $LAST_STEP * 100 / 100000" | bc 2>/dev/null)
        echo "  进度: ${PROGRESS}%"
    else
    echo "  等待训练数据..."
    fi
else
    echo "  训练日志未生成"
fi

echo ""

# 系统资源
echo "【系统资源】"
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
MEM_INFO=$(free -h | grep Mem)
DISK_INFO=$(df -h / | tail -1)

echo "  CPU 使用: ${CPU_USAGE}%"
echo "  内存: $(echo $MEM_INFO | awk '{print $3 "/" $2}')"
echo "  磁盘: $(echo $DISK_INFO | awk '{print $3 "/" $2 " (" $5 ")"}')"

echo ""

# Web 监控地址
echo "【Web 监控】"
if [ -n "$WEB_PROC" ]; then
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    echo "  本地访问: http://localhost:8081"
    echo "  局域网: http://${LOCAL_IP}:8081"
else
    echo "  Web 监控未启动"
fi

echo ""
echo "=========================================="
