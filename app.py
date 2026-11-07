
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import json
import logging
from typing import Dict, List, Optional
import ccxt.async_support as ccxt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Dashboard")


# ==========================================
# 1. TRADING BOT & STATE ENGINE
# ==========================================
class TradingState:

    def __init__(self):
        self.is_running = False
        self.symbol = "BTC/USDT"
        self.timeframe = "1m"
        self.cash = 10000.0
        self.initial_cash = 10000.0
        self.positions: Dict[str, dict] = {}
        self.trade_logs: List[dict] = []
        self.latest_candle: dict = {}


state = TradingState()
active_websockets: List[WebSocket] = []


async def broadcast_state():
    """Broadcasts live account, position, and latest candle to all open Chrome tabs."""
    if not active_websockets:
        return

    payload = {
        "status": "RUNNING" if state.is_running else "STOPPED",
        "cash": round(state.cash, 2),
        "total_equity": round(
            state.cash
            + sum(
                p["qty"] * state.latest_candle.get("close", p["entry_price"])
                for p in state.positions.values()
            ),
            2,
        ),
        "positions": list(state.positions.values()),
        "candle": state.latest_candle,
        "logs": state.trade_logs[-10:],
    }

    dead_sockets = []
    for ws in active_websockets:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            dead_sockets.append(ws)

    for ws in dead_sockets:
        if ws in active_websockets:
            active_websockets.remove(ws)


async def trading_bot_loop():
    """Async market data consumer and strategy executor."""
    exchange = ccxt.binance(
        {"enableRateLimit": True, "options": {"defaultType": "spot"}}
    )

    while True:
        if state.is_running:
            try:
                # 1. Fetch live OHLCV candle data
                ohlcv = await exchange.fetch_ohlcv(
                    state.symbol, timeframe=state.timeframe, limit=35
                )
                df = pd.DataFrame(
                    ohlcv,
                    columns=["time", "open", "high", "low", "close", "volume"],
                )

                # Format time for TradingView Lightweight Charts (epoch seconds)
                latest_raw = df.iloc[-1]
                state.latest_candle = {
                    "time": int(latest_raw["time"] / 1000),
                    "open": float(latest_raw["open"]),
                    "high": float(latest_raw["high"]),
                    "low": float(latest_raw["low"]),
                    "close": float(latest_raw["close"]),
                }

                # 2. Indicators: EMA (9/21) & ATR
                df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
                df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

                high_low = df["high"] - df["low"]
                high_close = (df["high"] - df["close"].shift()).abs()
                low_close = (df["low"] - df["close"].shift()).abs()
                df["atr"] = (
                    pd.concat([high_low, high_close, low_close], axis=1)
                    .max(axis=1)
                    .rolling(14)
                    .mean()
                )

                cur_price = float(df.iloc[-1]["close"])
                atr = (
                    float(df.iloc[-1]["atr"])
                    if not np.isnan(df.iloc[-1]["atr"])
                    else cur_price * 0.01
                )

                # Signals
                prev = df.iloc[-2]
                last = df.iloc[-1]
                bullish_cross = (prev["ema9"] <= prev["ema21"]) and (
                    last["ema9"] > last["ema21"]
                )
                bearish_cross = (prev["ema9"] >= prev["ema21"]) and (
                    last["ema9"] < last["ema21"]
                )

                # 3. Execution Logic
                has_position = state.symbol in state.positions

                if bullish_cross and not has_position:
                    # Risk 1.5% of equity per trade
                    risk_amount = state.cash * 0.015
                    sl_dist = atr * 1.5
                    qty = risk_amount / sl_dist
                    cost = qty * cur_price

                    if cost <= state.cash:
                        state.cash -= cost
                        stop_loss = cur_price - sl_dist
                        take_profit = cur_price + (sl_dist * 2.0)

                        state.positions[state.symbol] = {
                            "symbol": state.symbol,
                            "entry_price": cur_price,
                            "qty": round(qty, 4),
                            "stop_loss": round(stop_loss, 2),
                            "take_profit": round(take_profit, 2),
                        }
                        log = f"BUY {round(qty, 4)} {state.symbol} @ ${cur_price:.2f}"
                        state.trade_logs.append(
                            {
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "msg": log,
                                "type": "buy",
                            }
                        )
                        logger.info(log)

                elif has_position:
                    pos = state.positions[state.symbol]
                    # Check exits
                    hit_tp = cur_price >= pos["take_profit"]
                    hit_sl = cur_price <= pos["stop_loss"]

                    if hit_tp or hit_sl or bearish_cross:
                        exit_val = pos["qty"] * cur_price
                        state.cash += exit_val
                        pnl = exit_val - (pos["qty"] * pos["entry_price"])
                        reason = (
                            "TP"
                            if hit_tp
                            else ("SL" if hit_sl else "SIGNAL_EXIT")
                        )

                        log = f"SELL {state.symbol} ({reason}) @ ${cur_price:.2f} | PnL: ${pnl:.2f}"
                        state.trade_logs.append(
                            {
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "msg": log,
                                "type": "sell" if pnl >= 0 else "loss",
                            }
                        )
                        del state.positions[state.symbol]
                        logger.info(log)

                await broadcast_state()

            except Exception as e:
                logger.error(f"Trading loop error: {e}")

        await asyncio.sleep(2)  # Tick rate


# ==========================================
# 2. FASTAPI WEB SERVER
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start bot background task on server startup
    bot_task = asyncio.create_task(trading_bot_loop())
    yield
    bot_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/api/history")
async def get_history():
    """Initial chart backfill for Chrome."""
    exchange = ccxt.binance()
    ohlcv = await exchange.fetch_ohlcv(
        state.symbol, timeframe=state.timeframe, limit=100
    )
    await exchange.close()
    return [
        {
            "time": int(b[0] / 1000),
            "open": b[1],
            "high": b[2],
            "low": b[3],
            "close": b[4],
        }
        for b in ohlcv
    ]


@app.post("/api/toggle")
async def toggle_bot():
    state.is_running = not state.is_running
    await broadcast_state()
    return {"running": state.is_running}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    await broadcast_state()
    try:
        while True:
            await websocket.receive_text()  # Keep connection open
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)


# ==========================================
# 3. HTML/JS CHROME FRONTEND (Single Page)
# ==========================================
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Algo Trading Terminal</title>
    <!-- TradingView Lightweight Charts CDN -->
    <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        :root { --bg: #0f172a; --panel: #1e293b; --accent: #38bdf8; --border: #334155; --text: #f8fafc; --green: #22c55e; --red: #ef4444; }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        
        header { display: flex; justify-content: space-between; align-items: center; padding: 12px 20px; background: var(--panel); border-bottom: 1px solid var(--border); }
        .logo { font-size: 1.1rem; font-weight: 700; color: var(--accent); letter-spacing: 0.5px; }
        
        .status-badge { padding: 4px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
        .badge-stopped { background: rgba(239, 68, 68, 0.2); color: var(--red); border: 1px solid var(--red); }
        .badge-running { background: rgba(34, 197, 94, 0.2); color: var(--green); border: 1px solid var(--green); }
        
        button.btn { padding: 8px 16px; border: none; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .btn-start { background: var(--green); color: #fff; }
        .btn-stop { background: var(--red); color: #fff; }

        .container { display: grid; grid-template-columns: 1fr 340px; height: calc(100vh - 57px); }
        #chart { width: 100%; height: 100%; }

        .sidebar { background: var(--panel); border-left: 1px solid var(--border); display: flex; flex-direction: column; padding: 16px; gap: 16px; }
        .stat-card { background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
        .stat-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px; }
        .stat-value { font-size: 1.25rem; font-weight: 700; }

        .log-section { flex: 1; display: flex; flex-direction: column; min-height: 0; }
        .log-list { flex: 1; overflow-y: auto; background: #090d16; border-radius: 6px; border: 1px solid var(--border); padding: 8px; font-family: monospace; font-size: 0.75rem; }
        .log-item { margin-bottom: 6px; line-height: 1.3; }
        .log-buy { color: var(--accent); }
        .log-sell { color: var(--green); }
        .log-loss { color: var(--red); }
    </style>
</head>
<body>
    <header>
        <div style="display:flex; align-items:center; gap: 12px;">
            <div class="logo">TERMINAL // BTC-USDT</div>
            <span id="botBadge" class="status-badge badge-stopped">STOPPED</span>
        </div>
        <button id="toggleBtn" class="btn btn-start" onclick="toggleBot()">Start Engine</button>
    </header>

    <div class="container">
        <div id="chart"></div>
        <div class="sidebar">
            <div class="stat-card">
                <div class="stat-label">Total Portfolio Equity</div>
                <div class="stat-value" id="equityVal">$10,000.00</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Available Cash Balance</div>
                <div class="stat-value" id="cashVal">$10,000.00</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Open Position</div>
                <div id="positionStatus" style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Flat (No Trades Open)</div>
            </div>
            <div class="log-section">
                <div class="stat-label" style="margin-bottom: 6px;">Live Execution Stream</div>
                <div class="log-list" id="logs"></div>
            </div>
        </div>
    </div>

    <script>
        // 1. Initialize TradingView Lightweight Chart
        const chartContainer = document.getElementById('chart');
        const chart = LightweightCharts.createChart(chartContainer, {
            layout: { background: { color: '#0f172a' }, textColor: '#94a3b8' },
            grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
            crosshair: { mode: 0 },
            timeScale: { timeVisible: true, secondsVisible: false }
        });

        const candleSeries = chart.addCandlestickSeries({
            upColor: '#22c55e', downColor: '#ef4444',
            borderVisible: false, wickUpColor: '#22c55e', wickDownColor: '#ef4444'
        });

        window.addEventListener('resize', () => {
            chart.resize(chartContainer.clientWidth, chartContainer.clientHeight);
        });

        // 2. Fetch Historical Candlesticks on Load
        fetch('/api/history')
            .then(res => res.json())
            .then(data => {
                candleSeries.setData(data);
                chart.timeScale().fitContent();
            });

        // 3. Toggle Button
        async function toggleBot() {
            const res = await fetch('/api/toggle', { method: 'POST' });
            const data = await res.json();
            updateButtonState(data.running);
        }

        function updateButtonState(isRunning) {
            const btn = document.getElementById('toggleBtn');
            const badge = document.getElementById('botBadge');
            if (isRunning) {
                btn.className = 'btn btn-stop';
                btn.innerText = 'Stop Engine';
                badge.className = 'status-badge badge-running';
                badge.innerText = 'RUNNING';
            } else {
                btn.className = 'btn btn-start';
                btn.innerText = 'Start Engine';
                badge.className = 'status-badge badge-stopped';
                badge.innerText = 'STOPPED';
            }
        }

        // 4. WebSocket Live Updates
        const ws = new WebSocket(`ws://${location.host}/ws`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Update Candle in real time
            if (data.candle && data.candle.time) {
                candleSeries.update(data.candle);
            }

            // Sync Controls & Balances
            updateButtonState(data.status === 'RUNNING');
            document.getElementById('equityVal').innerText = '$' + data.total_equity.toLocaleString();
            document.getElementById('cashVal').innerText = '$' + data.cash.toLocaleString();

            // Render Positions
            const posEl = document.getElementById('positionStatus');
            if (data.positions && data.positions.length > 0) {
                const p = data.positions[0];
                posEl.innerHTML = `<span style="color:#38bdf8; font-weight:600;">LONG ${p.qty} BTC</span><br>Entry: $${p.entry_price} | SL: $${p.stop_loss}`;
            } else {
                posEl.innerHTML = `Flat (No Trades Open)`;
            }

            // Render Logs
            const logContainer = document.getElementById('logs');
            logContainer.innerHTML = data.logs.map(l => 
                `<div class="log-item log-${l.type}">[${l.time}] ${l.msg}</div>`
            ).join('');
        };
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return HTML_CONTENT


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
