#!/usr/bin/env python3
"""
HOT 训练数据推送器 — 将本地训练日志解析后推送到 Cloudflare D1

用法：
    CLOUDFLARE_API_TOKEN=xxx python pusher.py [--log train_current.log] [--interval 10]

环境变量：
    CLOUDFLARE_API_TOKEN — Cloudflare API Token（需要 D1 Write 权限）
"""

import re
import json
import os
import time
import argparse
import subprocess
from datetime import datetime

LOG_PATTERN = re.compile(
    r'\[step (\d+)\] train_loss=([\d.]+), learning_rate=([\d.]+), '
    r'annealing_beta=([\d.]+), steps_per_sec=([\d.]+), tokens_per_sec=([\d.]+)'
)
EVAL_PATTERN = re.compile(r'\[step (\d+)\] eval_loss=([\d.]+)')
TOTAL_PATTERN = re.compile(r'max_steps=(\d+)')
PARAMS_PATTERN = re.compile(r'模型参数量: ([\d.]+)M')
DEVICE_PATTERN = re.compile(r'设备: (\w+)')
COMPILE_PATTERN = re.compile(r'启用 torch.compile')
BATCH_PATTERN = re.compile(r'batch_size=(\d+)')
GRAD_ACCUM_PATTERN = re.compile(r'grad_accum=(\d+)')
START_PATTERN = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*开始训练')
CKPT_PATTERN = re.compile(r'检查点已保存.*step=(\d+)')

ACCOUNT_ID = "e82cfd2caf59231ca62061a3454df254"
D1_DATABASE_ID = "4636e359-0a9e-45ff-ad89-a4a0a2d9c649"
D1_API_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/d1/database/{D1_DATABASE_ID}/query"
MAX_POINTS = 500


def downsample(steps, *arrays, n=MAX_POINTS):
    total = len(steps)
    if total <= n:
        return [steps] + [list(a) for a in arrays]
    idxs = {0, total - 1}
    gap = total / n
    for i in range(n):
        idxs.add(int(i * gap))
    idxs = sorted(idxs)
    result = [[steps[i] for i in idxs]]
    for a in arrays:
        result.append([a[i] for i in idxs])
    return result


def parse_log(path: str) -> dict:
    steps, loss, lr, beta, sps, tps = [], [], [], [], [], []
    eval_steps, eval_loss = [], []
    total_steps, batch_size, grad_accum = 100000, 0, 0
    params_m, device, compiled = 0, 'unknown', False
    start_time, checkpoints = '', []

    try:
        with open(path) as f:
            for line in f:
                if m := TOTAL_PATTERN.search(line):
                    total_steps = int(m.group(1))
                if m := PARAMS_PATTERN.search(line):
                    params_m = float(m.group(1))
                if m := DEVICE_PATTERN.search(line):
                    device = m.group(1)
                if COMPILE_PATTERN.search(line):
                    compiled = True
                if m := BATCH_PATTERN.search(line):
                    batch_size = int(m.group(1))
                if m := GRAD_ACCUM_PATTERN.search(line):
                    grad_accum = int(m.group(1))
                if m := START_PATTERN.search(line):
                    start_time = m.group(1)
                if m := CKPT_PATTERN.search(line):
                    checkpoints.append(int(m.group(1)))
                if m := LOG_PATTERN.search(line):
                    steps.append(int(m.group(1)))
                    loss.append(float(m.group(2)))
                    lr.append(float(m.group(3)))
                    beta.append(float(m.group(4)))
                    sps.append(float(m.group(5)))
                    tps.append(float(m.group(6)))
                if m := EVAL_PATTERN.search(line):
                    eval_steps.append(int(m.group(1)))
                    eval_loss.append(float(m.group(2)))
    except FileNotFoundError:
        return None

    if not steps:
        return None

    w = 50
    loss_ma = []
    window_sum = 0.0
    for i, v in enumerate(loss):
        window_sum += v
        if i >= w:
            window_sum -= loss[i - w]
        loss_ma.append(window_sum / min(i + 1, w))

    cur_step = steps[-1]
    elapsed_hrs = 0.0
    if start_time:
        try:
            t0 = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            elapsed_hrs = (datetime.now() - t0).total_seconds() / 3600
        except Exception:
            pass

    ds_steps, ds_loss, ds_loss_ma, ds_lr, ds_beta, ds_sps, ds_tps = \
        downsample(steps, loss, loss_ma, lr, beta, sps, tps)

    return {
        'meta': {
            'total_steps': total_steps, 'params_m': params_m, 'device': device,
            'compiled': compiled, 'batch_size': batch_size, 'grad_accum': grad_accum,
            'current_step': cur_step, 'current_loss': loss[-1], 'min_loss': min(loss),
            'loss_ma': loss_ma[-1],
            'current_eval_loss': eval_loss[-1] if eval_loss else None,
            'min_eval_loss': min(eval_loss) if eval_loss else None,
            'current_tps': tps[-1],
            'avg_tps': sum(tps[-100:]) / min(len(tps), 100),
            'progress': cur_step / total_steps * 100, 'elapsed_hrs': elapsed_hrs,
            'checkpoints': checkpoints, 'start_time': start_time,
            'total_points': len(steps), 'chart_points': len(ds_steps),
        },
        'steps': ds_steps, 'loss': ds_loss, 'loss_ma': ds_loss_ma,
        'eval_steps': eval_steps, 'eval_loss': eval_loss,
        'lr': ds_lr, 'beta': ds_beta, 'steps_per_sec': ds_sps, 'tokens_per_sec': ds_tps,
    }


def d1_query(sql: str, params: list, api_token: str) -> bool:
    body = json.dumps({"sql": sql, "params": params})
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "15", "-X", "POST", D1_API_URL,
             "-H", f"Authorization: Bearer {api_token}",
             "-H", "Content-Type: application/json",
             "-d", body],
            capture_output=True, text=True, timeout=20,
        )
        resp = json.loads(result.stdout)
        return resp.get("success", False)
    except Exception as e:
        print(f"[pusher] D1 query failed: {e}")
        return False


def push_to_d1(data: dict, api_token: str) -> bool:
    ts = int(time.time())
    sql = """INSERT INTO snapshots (ts, meta, steps, loss, loss_ma, eval_steps, eval_loss, lr, beta, steps_per_sec, tokens_per_sec)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    params = [
        ts,
        json.dumps(data["meta"], separators=(",", ":")),
        json.dumps(data["steps"]), json.dumps(data["loss"]),
        json.dumps(data["loss_ma"]), json.dumps(data["eval_steps"]),
        json.dumps(data["eval_loss"]), json.dumps(data["lr"]),
        json.dumps(data["beta"]), json.dumps(data["steps_per_sec"]),
        json.dumps(data["tokens_per_sec"]),
    ]
    return d1_query(sql, params, api_token)


def cleanup_old(api_token: str, keep: int = 20):
    sql = f"DELETE FROM snapshots WHERE id NOT IN (SELECT id FROM snapshots ORDER BY ts DESC LIMIT {keep})"
    d1_query(sql, [], api_token)


def main():
    parser = argparse.ArgumentParser(description="HOT Training Data Pusher (D1)")
    parser.add_argument("--log", default="train_current.log", help="训练日志路径")
    parser.add_argument("--interval", type=int, default=10, help="推送间隔（秒）")
    args = parser.parse_args()

    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not api_token:
        print("错误: 请设置 CLOUDFLARE_API_TOKEN 环境变量")
        return

    print(f"[pusher] 启动数据推送器 (D1 via curl)")
    print(f"[pusher] 日志文件: {args.log}")
    print(f"[pusher] 推送间隔: {args.interval}s")
    print(f"[pusher] D1 database: {D1_DATABASE_ID}")

    last_step = -1
    push_count = 0
    while True:
        data = parse_log(args.log)
        if data:
            cur_step = data["meta"]["current_step"]
            if cur_step != last_step:
                ok = push_to_d1(data, api_token)
                status = "✓" if ok else "✗"
                print(f"[pusher] {status} step={cur_step} loss={data['meta']['current_loss']:.4f} progress={data['meta']['progress']:.1f}%")
                last_step = cur_step
                push_count += 1
                if push_count % 20 == 0:
                    cleanup_old(api_token)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
