#!/bin/bash
# 安装 HOT 训练监控器 systemd 服务

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="hot-train-monitor"
SERVICE_FILE="${SCRIPT_DIR}/${SERVICE_NAME}.service"

echo "=== 安装 HOT 训练监控器服务 ==="

# 检查是否为 root
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    echo "  sudo bash install_monitor.sh"
    exit 1
fi

# 复制服务文件
echo "复制服务文件..."
cp "$SERVICE_FILE" /etc/systemd/system/

# 重新加载 systemd
echo "重新加载 systemd..."
systemctl daemon-reload

# 启用服务
echo "启用服务..."
systemctl enable "$SERVICE_NAME"

echo ""
echo "=== 安装完成 ==="
echo ""
echo "可用命令："
echo "  启动监控: sudo systemctl start $SERVICE_NAME"
echo "  停止监控: sudo systemctl stop $SERVICE_NAME"
echo "  查看状态: sudo systemctl status $SERVICE_NAME"
echo "  查看日志: sudo journalctl -u $SERVICE_NAME -f"
echo "  禁用开机启动: sudo systemctl disable $SERVICE_NAME"
echo ""
echo "或直接运行（不使用 systemd）："
echo "  cd $SCRIPT_DIR"
echo "  python train_monitor.py"
