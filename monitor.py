#!/usr/bin/env python3
"""
HOT 训练监控服务（性能优化版）

优化点：
- 日志解析缓存：仅文件 mtime 变化时重新解析
- 数据降采样：图表最多 500 个点，大幅减少传输和渲染开销
- 增量更新：前端仅在数据变化时重绘
- Chart.js 零动画模式

用法：python monitor.py [--log train_10m.log] [--port 8080]
"""

import re
import json
import argparse
import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime

LOG_PATTERN = re.compile(
    r'\[step (\d+)\] train_loss=([\d.]+), learning_rate=([\d.]+), '
    r'annealing_beta=([\d.]+), steps_per_sec=([\d.]+), tokens_per_sec=([\d.]+)'
)
EVAL_PATTERN = re.compile(
    r'\[step (\d+)\] eval_loss=([\d.]+)'
)
TOTAL_PATTERN = re.compile(r'max_steps=(\d+)')
PARAMS_PATTERN = re.compile(r'模型参数量: ([\d.]+)M')
DEVICE_PATTERN = re.compile(r'设备: (\w+)')
COMPILE_PATTERN = re.compile(r'启用 torch.compile')
BATCH_PATTERN = re.compile(r'batch_size=(\d+)')
GRAD_ACCUM_PATTERN = re.compile(r'grad_accum=(\d+)')
START_PATTERN = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*开始训练')
CKPT_PATTERN = re.compile(r'检查点已保存.*step=(\d+)')

MAX_POINTS = 500  # 图表最大数据点数


def downsample(steps, *arrays, n=MAX_POINTS):
    """均匀降采样，保留首尾点"""
    total = len(steps)
    if total <= n:
        return [steps] + [list(a) for a in arrays]
    idxs = set()
    idxs.add(0)
    idxs.add(total - 1)
    gap = total / n
    for i in range(n):
        idxs.add(int(i * gap))
    idxs = sorted(idxs)
    result = [[steps[i] for i in idxs]]
    for a in arrays:
        result.append([a[i] for i in idxs])
    return result


class LogCache:
    """带缓存的日志解析器，仅文件变化时重新解析"""

    def __init__(self):
        self._cache = None
        self._mtime = 0
        self._size = 0
        self._lock = threading.Lock()

    def get(self, path: str) -> dict:
        try:
            st = os.stat(path)
            mtime, size = st.st_mtime, st.st_size
        except FileNotFoundError:
            return self._empty()

        with self._lock:
            if self._cache and mtime == self._mtime and size == self._size:
                return self._cache
            self._mtime, self._size = mtime, size
            self._cache = self._parse(path)
            return self._cache

    @staticmethod
    def _parse(path: str) -> dict:
        steps, loss, lr, beta, sps, tps = [], [], [], [], [], []
        eval_steps, eval_loss = [], []
        total_steps, batch_size, grad_accum = 100000, 0, 0
        params_m, device, compiled = 0, 'unknown', False
        start_time, checkpoints = '', []

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

        if not steps:
            return {'meta': {}, 'steps': [], 'loss': [], 'loss_ma': [],
                    'lr': [], 'beta': [], 'steps_per_sec': [], 'tokens_per_sec': []}

        # loss 移动平均（窗口 50，增量计算）
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

        # 降采样图表数据
        ds_steps, ds_loss, ds_loss_ma, ds_lr, ds_beta, ds_sps, ds_tps = \
            downsample(steps, loss, loss_ma, lr, beta, sps, tps)
        # eval loss 点较少，不需要降采样
        ds_eval_steps, ds_eval_loss = eval_steps, eval_loss

        return {
            'meta': {
                'total_steps': total_steps,
                'params_m': params_m,
                'device': device,
                'compiled': compiled,
                'batch_size': batch_size,
                'grad_accum': grad_accum,
                'current_step': cur_step,
                'current_loss': loss[-1],
                'min_loss': min(loss),
                'loss_ma': loss_ma[-1],
                'current_eval_loss': eval_loss[-1] if eval_loss else None,
                'min_eval_loss': min(eval_loss) if eval_loss else None,
                'current_tps': tps[-1],
                'avg_tps': sum(tps[-100:]) / min(len(tps), 100),
                'progress': cur_step / total_steps * 100,
                'elapsed_hrs': elapsed_hrs,
                'checkpoints': checkpoints,
                'start_time': start_time,
                'total_points': len(steps),
                'chart_points': len(ds_steps),
            },
            'steps': ds_steps,
            'loss': ds_loss,
            'loss_ma': ds_loss_ma,
            'eval_steps': ds_eval_steps,
            'eval_loss': ds_eval_loss,
            'lr': ds_lr,
            'beta': ds_beta,
            'steps_per_sec': ds_sps,
            'tokens_per_sec': ds_tps,
        }

    @staticmethod
    def _empty():
        return {'meta': {'progress': 0, 'current_step': 0, 'total_steps': 100000},
                'steps': [], 'loss': [], 'loss_ma': [], 'lr': [], 'beta': [],
                'steps_per_sec': [], 'tokens_per_sec': []}


_cache = LogCache()


class Handler(SimpleHTTPRequestHandler):
    log_path = 'train_10m.log'

    def do_GET(self):
        if self.path == '/api/data':
            data = _cache.get(self.log_path)
            payload = json.dumps(data, separators=(',', ':')).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass


HTML = r'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HOT Training Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
:root{--bg:#0a0e1a;--card:#111827;--card-h:#1a2332;--border:#1e293b;--text:#e2e8f0;--dim:#64748b;--muted:#475569;--blue:#3b82f6;--green:#10b981;--amber:#f59e0b;--purple:#8b5cf6;--pink:#ec4899;--red:#ef4444;--cyan:#06b6d4}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',-apple-system,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}

.hdr{display:flex;align-items:center;justify-content:space-between;padding:14px 28px;border-bottom:1px solid var(--border);backdrop-filter:blur(12px);background:rgba(10,14,26,.85);position:sticky;top:0;z-index:100}
.hdr-l{display:flex;align-items:center;gap:14px}
.logo{font-size:22px;font-weight:800;background:linear-gradient(135deg,var(--amber),var(--pink));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.logo-s{font-size:10px;color:var(--dim);font-weight:600;letter-spacing:1.2px;text-transform:uppercase}
.hdr-r{display:flex;align-items:center;gap:10px}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:600;padding:4px 12px;border-radius:999px;background:var(--card);border:1px solid var(--border)}
.dot{width:6px;height:6px;border-radius:50%;display:inline-block}
.dot.on{background:var(--green);animation:pulse 2s infinite}
.dot.off{background:var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

.pbar{padding:0 28px;margin-top:-1px}
.pbar-t{height:3px;background:var(--border);border-radius:2px;overflow:hidden}
.pbar-f{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--blue),var(--purple),var(--pink));transition:width .6s ease}

.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;padding:16px 28px}
@media(max-width:1200px){.stats{grid-template-columns:repeat(3,1fr)}}
@media(max-width:600px){.stats{grid-template-columns:repeat(2,1fr)}}
.sc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 16px;transition:border-color .15s}
.sc:hover{border-color:var(--blue)}
.sc .ic{font-size:13px;margin-bottom:6px;opacity:.5}
.sc .lb{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.8px;font-weight:600}
.sc .vl{font-size:24px;font-weight:800;margin-top:3px;font-variant-numeric:tabular-nums;line-height:1.1}
.sc .sb{font-size:10px;color:var(--muted);margin-top:3px;font-weight:500}

.charts{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:0 28px 16px}
@media(max-width:900px){.charts{grid-template-columns:1fr}}
.cc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px;height:220px;overflow:hidden}
.cc h3{font-size:11px;color:var(--dim);font-weight:600;text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px;display:flex;align-items:center;gap:7px}
.cc h3 .cd{width:7px;height:7px;border-radius:50%}
.cc canvas{height:170px!important}

.ibar{display:flex;gap:10px;padding:0 28px 16px;flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;gap:5px;font-size:10px;color:var(--dim);font-weight:500;padding:5px 12px;background:var(--card);border:1px solid var(--border);border-radius:999px}
.chip b{color:var(--text);font-weight:700}

.ft{text-align:center;padding:14px;font-size:10px;color:var(--muted);border-top:1px solid var(--border)}
</style>
</head>
<body>

<div class="hdr">
  <div class="hdr-l"><div><div class="logo">HOT</div><div class="logo-s">Training Monitor</div></div></div>
  <div class="hdr-r">
    <span class="badge"><span class="dot on" id="dot"></span><span id="st">connecting</span></span>
    <span class="badge" id="dev">—</span>
  </div>
</div>

<div class="pbar"><div class="pbar-t"><div class="pbar-f" id="bar" style="width:0"></div></div></div>

<div class="stats">
  <div class="sc"><div class="ic">📊</div><div class="lb">Progress</div><div class="vl" id="s-prog">—</div><div class="sb" id="s-progs"></div></div>
  <div class="sc"><div class="ic">📉</div><div class="lb">Loss</div><div class="vl" id="s-loss">—</div><div class="sb" id="s-losss"></div><div class="sb" id="s-eval" style="color:var(--cyan);margin-top:2px"></div></div>
  <div class="sc"><div class="ic">🚀</div><div class="lb">Throughput</div><div class="vl" id="s-tps">—</div><div class="sb">tok/s</div></div>
  <div class="sc"><div class="ic">⚡</div><div class="lb">Speed</div><div class="vl" id="s-sps">—</div><div class="sb">step/s</div></div>
  <div class="sc"><div class="ic">⏱️</div><div class="lb">ETA</div><div class="vl" id="s-eta">—</div><div class="sb" id="s-etas"></div></div>
  <div class="sc"><div class="ic">📈</div><div class="lb">LR</div><div class="vl" id="s-lr">—</div><div class="sb" id="s-lrs"></div></div>
</div>

<div class="ibar" id="ibar"></div>

<div class="charts">
  <div class="cc"><h3><span class="cd" style="background:var(--amber)"></span>Train Loss <span class="cd" style="background:var(--pink);margin-left:8px"></span>MA <span class="cd" style="background:var(--cyan);margin-left:8px"></span>Eval</h3><canvas id="cLoss"></canvas></div>
  <div class="cc"><h3><span class="cd" style="background:var(--green)"></span>Throughput</h3><canvas id="cTps"></canvas></div>
  <div class="cc"><h3><span class="cd" style="background:var(--blue)"></span>Learning Rate</h3><canvas id="cLr"></canvas></div>
  <div class="cc"><h3><span class="cd" style="background:var(--purple)"></span>Annealing β</h3><canvas id="cBeta"></canvas></div>
</div>

<div class="ft">HOT — Harmonic Oscillator Transformer · <span id="ftime"></span></div>

<script>
const C={blue:'#3b82f6',green:'#10b981',amber:'#f59e0b',purple:'#8b5cf6',pink:'#ec4899'};
const $=id=>document.getElementById(id);
let prevHash='';

function mkGrad(ctx,c){const g=ctx.createLinearGradient(0,0,0,200);g.addColorStop(0,c+'30');g.addColorStop(1,c+'00');return g}
function mkC(id,c){const ctx=$(id).getContext('2d');return new Chart(ctx,{type:'line',data:{labels:[],datasets:[{data:[],borderColor:c,borderWidth:1.5,backgroundColor:mkGrad(ctx,c),fill:true,tension:.4,pointRadius:0,pointHitRadius:5}]},options:{responsive:true,maintainAspectRatio:false,animation:false,interaction:{mode:'index',intersect:false},scales:{x:{ticks:{color:'#1e293b',maxTicksLimit:6,font:{size:9,monospace:true}},grid:{display:false}},y:{ticks:{color:'#334155',font:{size:9}},grid:{color:'#1e293b44'}}},plugins:{legend:{display:false},tooltip:{backgroundColor:'#1e293b',titleColor:'#e2e8f0',bodyColor:'#94a3b8',borderColor:'#334155',borderWidth:1,padding:8,cornerRadius:6,titleFont:{size:11},bodyFont:{size:10,family:'monospace'}}}}})}
const cLoss=mkC('cLoss',C.amber),cTps=mkC('cTps',C.green),cLr=mkC('cLr',C.blue),cBeta=mkC('cBeta',C.purple);

// loss MA 第二条线
cLoss.data.datasets.push({data:[],borderColor:C.pink,borderWidth:1.5,backgroundColor:'transparent',fill:false,tension:.4,pointRadius:0});
// eval loss 散点（红色菱形标记）
cLoss.data.datasets.push({data:[],borderColor:'#06b6d4',backgroundColor:'#06b6d4',pointRadius:4,pointStyle:'rectRot',showLine:false,fill:false});

function fmtN(n){return n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'k':n.toFixed(1)}
function etaF(h){return h<0?'—':h<1?Math.round(h*60)+'m':h.toFixed(1)+'h'}

function setData(chart,labels,data){chart.data.labels=labels;chart.data.datasets[0].data=data}

async function tick(){
  try{
    const r=await fetch('/api/data');
    const d=await r.json();
    const m=d.meta,s=d.steps;
    if(!s.length)return;

    // 数据哈希，避免无变化时重绘
    const hash=m.current_step+'_'+m.current_loss.toFixed(4);
    const changed=hash!==prevHash;
    prevHash=hash;

    // 状态栏（每次都更新，很轻量）
    $('dot').className='dot on';$('st').textContent='LIVE';
    $('dev').textContent=m.device+(m.compiled?' + compiled':'');
    $('s-prog').textContent=m.progress.toFixed(1)+'%';
    $('s-progs').textContent=fmtN(m.current_step)+' / '+fmtN(m.total_steps);
    $('s-loss').textContent=m.current_loss.toFixed(4);
    $('s-losss').textContent='MA: '+m.loss_ma.toFixed(4)+' · min: '+m.min_loss.toFixed(4);
    if(m.current_eval_loss!==null){
      $('s-eval').textContent='eval: '+m.current_eval_loss.toFixed(4)+' · min: '+m.min_eval_loss.toFixed(4);
    }
    $('s-tps').textContent=fmtN(m.current_tps);
    $('s-sps').textContent=d.steps_per_sec.at(-1).toFixed(2);
    const rem=(m.total_steps-m.current_step)/d.steps_per_sec.at(-1)/3600;
    $('s-eta').textContent=etaF(rem);
    $('s-etas').textContent='elapsed: '+m.elapsed_hrs.toFixed(1)+'h';
    $('s-lr').textContent=d.lr.at(-1).toExponential(1);
    $('s-lrs').textContent='init: '+Math.max(...d.lr).toExponential(1);
    $('bar').style.width=m.progress+'%';
    $('ftime').textContent=new Date().toLocaleTimeString();

    // info chips
    const ch=[];
    if(m.batch_size)ch.push('batch <b>'+m.batch_size+'</b>');
    if(m.grad_accum)ch.push('accum <b>'+m.grad_accum+'</b>');
    ch.push('params <b>'+m.params_m.toFixed(1)+'M</b>');
    ch.push('β <b>'+d.beta.at(-1).toFixed(2)+'</b>');
    if(m.checkpoints.length)ch.push('ckpt <b>'+fmtN(m.checkpoints.at(-1))+'</b>');
    ch.push('chart <b>'+m.chart_points+'/'+m.total_points+'</b> pts');
    $('ibar').innerHTML=ch.map(c=>'<span class="chip">'+c+'</span>').join('');

    // 图表仅在数据变化时更新
    if(!changed)return;
    setData(cLoss,s,d.loss);cLoss.data.datasets[1].data=d.loss_ma;
    // eval loss: 需要对齐到 chart 的 step labels
    if(d.eval_steps&&d.eval_steps.length){
      const evalMap={};d.eval_steps.forEach((s,i)=>evalMap[s]=d.eval_loss[i]);
      cLoss.data.datasets[2].data=s.map(st=>evalMap[st]??null);
    }
    cLoss.update('none');
    setData(cTps,s,d.tokens_per_sec);cTps.update('none');
    setData(cLr,s,d.lr);cLr.update('none');
    setData(cBeta,s,d.beta);cBeta.update('none');

  }catch(e){$('dot').className='dot off';$('st').textContent='OFFLINE'}
}

tick();setInterval(tick,5000);
</script>
</body>
</html>'''


def find_latest_log():
    """查找最新的训练日志"""
    candidates = ['train_final.log', 'train_current.log', 'train_10m.log']
    for log in candidates:
        if os.path.exists(log) and not os.path.islink(log):
            return log
    # 回退：查找最新的 train*.log
    import glob
    logs = glob.glob('train*.log')
    if logs:
        return max(logs, key=os.path.getmtime)
    return 'train_final.log'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=None, help='日志文件路径（默认自动检测）')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    log_path = args.log or find_latest_log()
    Handler.log_path = log_path
    server = HTTPServer(('0.0.0.0', args.port), Handler)
    print(f'监控面板: http://localhost:{args.port}')
    print(f'日志文件: {log_path}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n已停止')
