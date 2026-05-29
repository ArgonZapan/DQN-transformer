import { useEffect, useMemo, useRef, useState } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';

const FONT = 'IBM Plex Mono,monospace';
const INTERVALS = ['15m', '1h', '4h', '1d'];
const TF_HOURS = { '15m': 0.25, '1h': 1, '4h': 4, '1d': 24 };
const KRONOS_MAX_PRED_LEN = 1024;  // matches backend cap

const STORAGE_KEY = 'kronos.settings.v1';

// Defaults — overridden by anything found in localStorage.
const DEFAULT_SETTINGS = {
  symbol: 'BTCUSDT',
  interval: '1h',
  lookback: 400,
  days: 7,
  T: 1.0,
  topP: 0.9,
  nSimulations: 100,
  model: 'mini',
};

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function saveSettings(s) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(s)); } catch { /* ignore */ }
}

function predLenFromDays(days, interval) {
  const h = TF_HOURS[interval] || 1;
  const raw = Math.max(1, Math.round((days * 24) / h));
  return Math.min(KRONOS_MAX_PRED_LEN, raw);
}

// Cap days so the resulting pred_len stays within the backend KRONOS_MAX_PRED_LEN.
function maxDaysForInterval(interval) {
  const h = TF_HOURS[interval] || 1;
  return Math.max(1, Math.floor((KRONOS_MAX_PRED_LEN * h) / 24));
}

// Chart with manual candle injection (no auto-fetch).
function KronosChart({ candles, generatedRanges, height = 460 }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  const overlayRefs = useRef([]);

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const chart = createChart(el, {
      layout: { background: { color: 'transparent' }, textColor: '#8890a4' },
      grid: { vertLines: { color: '#181b26' }, horzLines: { color: '#181b26' } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#1e2230' },
      timeScale: { borderColor: '#1e2230', timeVisible: true, secondsVisible: false },
      width: el.clientWidth,
      height,
    });
    const series = chart.addCandlestickSeries({
      upColor: '#22c55e', downColor: '#ef4444',
      borderUpColor: '#22c55e', borderDownColor: '#ef4444',
      wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    });
    chartRef.current = chart;
    seriesRef.current = series;
    const ro = new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth }));
    ro.observe(el);
    return () => { ro.disconnect(); chart.remove(); chartRef.current = null; seriesRef.current = null; };
  }, [height]);

  useEffect(() => {
    const series = seriesRef.current;
    const chart = chartRef.current;
    if (!series || !chart) return;
    series.setData(candles);

    // Clear previous overlays.
    overlayRefs.current.forEach(s => { try { chart.removeSeries(s); } catch { /* noop */ } });
    overlayRefs.current = [];

    // Color each generated batch with a distinct hue so users can see segments.
    const palette = [
      'rgba(59,130,246,0.55)',  // blue
      'rgba(168,85,247,0.55)',  // purple
      'rgba(236,72,153,0.55)',  // pink
      'rgba(249,115,22,0.55)',  // orange
      'rgba(20,184,166,0.55)',  // teal
    ];
    // Candles are sorted by time → use binary search to slice each range
    // in O(log N) instead of scanning the full array per overlay.
    const lowerBound = (t) => {
      let lo = 0, hi = candles.length;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (candles[mid].time < t) lo = mid + 1; else hi = mid;
      }
      return lo;
    };
    const upperBound = (t) => {
      let lo = 0, hi = candles.length;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (candles[mid].time <= t) lo = mid + 1; else hi = mid;
      }
      return lo;
    };
    generatedRanges.forEach((rng, i) => {
      const a = lowerBound(rng.from);
      const b = upperBound(rng.to);
      if (b <= a) return;
      const color = palette[i % palette.length];
      const line = chart.addLineSeries({
        color, lineWidth: 1, priceLineVisible: false,
        lastValueVisible: false, crosshairMarkerVisible: false,
      });
      const data = new Array(b - a);
      for (let j = a; j < b; j++) data[j - a] = { time: candles[j].time, value: candles[j].close };
      line.setData(data);
      overlayRefs.current.push(line);
    });

    if (candles.length) {
      const tail = Math.min(candles.length, 250);
      chart.timeScale().setVisibleRange({
        from: candles[candles.length - tail].time,
        to: candles[candles.length - 1].time,
      });
    }
  }, [candles, generatedRanges]);

  return (
    <div style={{ background: '#0c0d11', borderRadius: 6, border: '1px solid #1e2230', overflow: 'hidden' }}>
      <div ref={containerRef} style={{ width: '100%', height }} />
    </div>
  );
}

function rowsToCandles(rows) {
  return rows.map(k => ({
    time: Math.floor(k[0] / 1000),
    open: +k[1], high: +k[2], low: +k[3], close: +k[4], volume: +k[5],
  }));
}

function candlesToRows(candles) {
  // Re-pack to Binance-style 12-col rows for the backend.
  return candles.map(c => [
    c.time * 1000, c.open, c.high, c.low, c.close, c.volume,
    c.time * 1000, 0, 0, 0, 0, '0',
  ]);
}

const labelStyle = { fontSize: 10, color: '#505870', fontFamily: FONT, textTransform: 'uppercase', letterSpacing: '0.06em' };
const inputStyle = { background: '#111318', color: '#e2e4ea', border: '1px solid #1e2230', borderRadius: 4, padding: '4px 8px', fontSize: 11, fontFamily: FONT, width: 80 };

export default function KronosSection() {
  const [settings, setSettings] = useState(loadSettings);
  const [symbols, setSymbols] = useState(['BTCUSDT']);
  const [models, setModels] = useState([
    { name: 'mini', max_context: 2048, loaded: false },
    { name: 'small', max_context: 512, loaded: false },
    { name: 'base', max_context: 512, loaded: false },
  ]);
  const [history, setHistory] = useState([]);          // base history candles (chart units)
  const [generated, setGenerated] = useState([]);      // appended forecast candles
  const [generatedRanges, setGeneratedRanges] = useState([]);  // [{from,to}, ...] per click
  const [loadingHist, setLoadingHist] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [bruteforcing, setBruteforcing] = useState(false);
  const [bruteforceResult, setBruteforceResult] = useState(null);
  const [bruteforceProgress, setBruteforceProgress] = useState(null); // { progress, total }

  // Persist settings on every change.
  useEffect(() => { saveSettings(settings); }, [settings]);

  // Load symbols from /api/config (best-effort).
  useEffect(() => {
    fetch('/api/config').then(r => r.json()).then(c => {
      if (c?.symbols?.length) setSymbols(c.symbols);
    }).catch(() => {});
  }, []);

  // Available Kronos models (refreshes after generate/bruteforce so loaded flags stay current).
  const refreshModels = () => {
    fetch('/api/kronos/models').then(r => r.json()).then(d => {
      if (d?.models) setModels(d.models);
    }).catch(() => {});
  };
  useEffect(() => { refreshModels(); }, []);

  // (Re)load history whenever symbol/interval/lookback changes.
  useEffect(() => {
    let cancelled = false;
    setLoadingHist(true); setError(null);
    setGenerated([]); setGeneratedRanges([]);
    const url = `/api/klines?symbol=${settings.symbol}&interval=${settings.interval}&limit=${settings.lookback}`;
    fetch(url)
      .then(r => { if (!r.ok) throw new Error('klines ' + r.status); return r.json(); })
      .then(rows => { if (!cancelled) { setHistory(rowsToCandles(rows)); setLoadingHist(false); } })
      .catch(e => { if (!cancelled) { setError(e.message); setLoadingHist(false); } });
    return () => { cancelled = true; };
  }, [settings.symbol, settings.interval, settings.lookback]);

  const allCandles = useMemo(() => [...history, ...generated], [history, generated]);

  const update = (patch) => setSettings(s => ({ ...s, ...patch }));

  async function onGenerate() {
    if (!history.length || generating) return;
    setGenerating(true); setError(null);
    try {
      const predLen = predLenFromDays(settings.days, settings.interval);
      // Send full known history (base + previously-generated) so the model
      // continues from the latest tip.
      const histRows = candlesToRows(allCandles);
      const seed = (Math.random() * 2 ** 31) | 0;
      const r = await fetch('/api/kronos/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          history: histRows,
          pred_len: predLen,
          T: settings.T,
          top_p: settings.topP,
          seed,
          model: settings.model,
        }),
      });
      if (!r.ok) throw new Error('generate ' + r.status + ': ' + await r.text());
      const data = await r.json();
      const newCandles = rowsToCandles(data.candles);
      if (!newCandles.length) throw new Error('empty response');
      setGenerated(g => [...g, ...newCandles]);
      setGeneratedRanges(rng => [...rng, { from: newCandles[0].time, to: newCandles[newCandles.length - 1].time }]);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setGenerating(false);
      refreshModels();
    }
  }

  function onReset() {
    setGenerated([]);
    setGeneratedRanges([]);
    setBruteforceResult(null);
  }

  async function onBruteforce() {
    if (!history.length || bruteforcing) return;
    setBruteforcing(true); setError(null); setBruteforceResult(null);
    setBruteforceProgress({ progress: 0, total: settings.nSimulations });
    try {
      const predLen = predLenFromDays(settings.days, settings.interval);
      const r = await fetch('/api/kronos/bruteforce', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          history: candlesToRows(history),
          pred_len: predLen,
          n_simulations: settings.nSimulations,
          T: settings.T,
          top_p: settings.topP,
          model: settings.model,
        }),
      });
      if (!r.ok) throw new Error('bruteforce ' + r.status + ': ' + await r.text());

      // Stream NDJSON: progress lines, then a final result/error line.
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let finalResult = null;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          let msg;
          try { msg = JSON.parse(line); } catch { continue; }
          if (msg.error) throw new Error(msg.error);
          if (msg.result) finalResult = msg.result;
          else if (typeof msg.progress === 'number') setBruteforceProgress(msg);
        }
      }
      if (!finalResult) throw new Error('bruteforce: no result received');
      setBruteforceResult(finalResult);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBruteforcing(false);
      setBruteforceProgress(null);
      refreshModels();
    }
  }

  const predLen = predLenFromDays(settings.days, settings.interval);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Top control bar */}
      <div style={{ background: '#111318', border: '1px solid #1e2230', borderRadius: 8, padding: '12px 16px', display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-end' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={labelStyle}>Symbol</span>
          <select style={{ ...inputStyle, width: 120 }} value={settings.symbol} onChange={e => update({ symbol: e.target.value })}>
            {symbols.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={labelStyle}>Interval</span>
          <select style={{ ...inputStyle, width: 80 }} value={settings.interval} onChange={e => update({ interval: e.target.value })}>
            {INTERVALS.map(iv => <option key={iv} value={iv}>{iv}</option>)}
          </select>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={labelStyle}>Model</span>
          <select style={{ ...inputStyle, width: 130 }} value={settings.model} onChange={e => update({ model: e.target.value })}>
            {models.map(m => (
              <option key={m.name} value={m.name}>
                {m.name}{m.loaded ? ' ✓' : ''} (ctx {m.max_context})
              </option>
            ))}
          </select>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={labelStyle}>Lookback (candles)</span>
          <input type="number" min={64} max={800} step={32} style={inputStyle}
            value={settings.lookback}
            onChange={e => update({ lookback: Math.max(64, Math.min(800, +e.target.value || 400)) })} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={labelStyle}>Days to gen</span>
          <input type="number" min={1} max={maxDaysForInterval(settings.interval)} step={1} style={inputStyle}
            value={settings.days}
            onChange={e => {
              const cap = maxDaysForInterval(settings.interval);
              update({ days: Math.max(1, Math.min(cap, +e.target.value || 7)) });
            }} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={labelStyle}>T (temp)</span>
          <input type="number" min={0.1} max={2.0} step={0.05} style={inputStyle}
            value={settings.T}
            onChange={e => update({ T: Math.max(0.1, Math.min(2.0, +e.target.value || 1.0)) })} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={labelStyle}>top_p</span>
          <input type="number" min={0.1} max={1.0} step={0.05} style={inputStyle}
            value={settings.topP}
            onChange={e => update({ topP: Math.max(0.1, Math.min(1.0, +e.target.value || 0.9)) })} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={labelStyle}>Simulations</span>
          <input type="number" min={1} max={500} step={10} style={inputStyle}
            value={settings.nSimulations}
            onChange={e => update({ nSimulations: Math.max(1, Math.min(500, +e.target.value || 100)) })} />
        </div>
        <div style={{ marginLeft: 'auto', fontSize: 10, color: '#3a3f52', fontFamily: FONT }}>
          pred_len = {predLen} candles
        </div>
      </div>

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <button onClick={onGenerate} disabled={generating || loadingHist}
          style={{
            background: generating ? '#1e2230' : '#3b82f6', color: '#fff',
            border: 'none', borderRadius: 4, padding: '8px 16px', fontSize: 11, fontFamily: FONT,
            fontWeight: 600, cursor: (generating || loadingHist) ? 'wait' : 'pointer',
            letterSpacing: '0.04em', textTransform: 'uppercase',
          }}>
          {generating ? 'Generating…' : `Generate ${settings.days}d`}
        </button>
        <button onClick={onReset} disabled={!generated.length}
          style={{
            background: 'transparent', color: '#8890a4',
            border: '1px solid #1e2230', borderRadius: 4, padding: '8px 14px', fontSize: 11, fontFamily: FONT,
            cursor: generated.length ? 'pointer' : 'not-allowed', letterSpacing: '0.04em', textTransform: 'uppercase',
          }}>Reset</button>
        <button onClick={onBruteforce} disabled={bruteforcing || loadingHist}
          style={{
            background: bruteforcing ? '#1e2230' : '#a855f7', color: '#fff',
            border: 'none', borderRadius: 4, padding: '8px 16px', fontSize: 11, fontFamily: FONT,
            fontWeight: 600, cursor: (bruteforcing || loadingHist) ? 'wait' : 'pointer',
            letterSpacing: '0.04em', textTransform: 'uppercase',
          }}>
          {bruteforcing
            ? (bruteforceProgress
                ? `Bruteforcing… ${bruteforceProgress.progress}/${bruteforceProgress.total}`
                : `Bruteforcing… (${settings.nSimulations} sims)`)
            : `Bruteforce (${settings.nSimulations} sims)`}
        </button>
        {bruteforcing && bruteforceProgress && bruteforceProgress.total > 0 && (
          <div style={{ width: 160, height: 6, background: '#1e2230', borderRadius: 3, overflow: 'hidden' }}>
            <div style={{
              width: `${Math.round(100 * bruteforceProgress.progress / bruteforceProgress.total)}%`,
              height: '100%', background: '#a855f7', transition: 'width 200ms linear',
            }} />
          </div>
        )}
        <span style={{ fontSize: 10, color: '#3a3f52', fontFamily: FONT }}>
          {generatedRanges.length} batch{generatedRanges.length === 1 ? '' : 'es'} generated
        </span>
        {error && (
          <span style={{ marginLeft: 'auto', fontSize: 11, color: '#ef4444', fontFamily: FONT }}>
            {error}
          </span>
        )}
      </div>

      {/* Chart */}
      {loadingHist ? (
        <div style={{ color: '#505870', fontFamily: FONT, fontSize: 12, padding: 20 }}>Loading history…</div>
      ) : (
        <KronosChart candles={allCandles} generatedRanges={generatedRanges} />
      )}

      {/* Bruteforce results */}
      {bruteforceResult && (
        <BruteforceTable result={bruteforceResult} interval={settings.interval} />
      )}
    </div>
  );
}

function BruteforceTable({ result, interval }) {
  const combos = result.combos || [];
  const top = combos.slice(0, 50);
  const meta = result.meta || {};
  const hours = TF_HOURS[interval] || 1;

  const sqnColor = (v) => v >= 2 ? '#22c55e' : v >= 1 ? '#f59e0b' : '#ef4444';
  const th = (label) => (
    <th style={{ padding: '6px 8px', textAlign: 'right', color: '#505870', fontSize: 10, fontFamily: FONT, fontWeight: 400, borderBottom: '1px solid #1e2230', whiteSpace: 'nowrap', background: '#0f1014' }}>{label}</th>
  );
  const td = (content, extra = {}) => (
    <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: FONT, fontSize: 11, borderBottom: '1px solid #181b26', color: '#8890a4', ...extra }}>{content}</td>
  );

  return (
    <div style={{ background: '#111318', border: '1px solid #1e2230', borderRadius: 8, overflow: 'hidden' }}>
      <div style={{ padding: '10px 14px', borderBottom: '1px solid #1e2230', display: 'flex', gap: 16, fontSize: 10, color: '#505870', fontFamily: FONT, flexWrap: 'wrap' }}>
        <span><b style={{ color: '#e2e4ea' }}>Bruteforce results</b></span>
        <span>Sims: {meta.n_simulations}</span>
        <span>pred_len: {meta.pred_len}</span>
        <span>T={meta.T}, top_p={meta.top_p}</span>
        <span>last close: {meta.last_close}</span>
        <span style={{ marginLeft: 'auto' }}>Showing top {top.length} of {combos.length}</span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 800 }}>
          <thead>
            <tr>
              {th('#')}{th('Dir')}{th('Hz (h)')}{th('Thr%')}
              {th('N')}{th('P(Win)')}{th('Avg+')}{th('Avg-')}
              {th('Mean%')}{th('EV%')}{th('SQN')}{th('Kelly')}
              {th('TP')}{th('SL')}{th('HZ')}
            </tr>
          </thead>
          <tbody>
            {top.map((c, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? '#0b0c0f' : '#0e0f13' }}>
                {td(i + 1, { color: '#505870' })}
                {td(c.direction.toUpperCase(), { color: c.direction === 'long' ? '#22c55e' : '#ef4444', fontWeight: 600 })}
                {td((c.horizon_bars * hours) + 'h')}
                {td(c.threshold_pct + '%')}
                {td(c.n_trades)}
                {td((c.p_win * 100).toFixed(1) + '%')}
                {td(c.avg_win > 0 ? '+' + c.avg_win.toFixed(2) + '%' : '—',
                    { color: c.avg_win > 0 ? '#22c55e' : '#3a3f52' })}
                {td(c.avg_loss > 0 ? '-' + c.avg_loss.toFixed(2) + '%' : '—',
                    { color: c.avg_loss > 0 ? '#ef4444' : '#3a3f52' })}
                {td((c.mean_return >= 0 ? '+' : '') + c.mean_return.toFixed(2) + '%', { color: c.mean_return >= 0 ? '#22c55e' : '#ef4444', fontWeight: 600 })}
                {td((c.ev >= 0 ? '+' : '') + c.ev.toFixed(2) + '%', { color: c.ev >= 0 ? '#22c55e' : '#ef4444' })}
                {td(c.sqn, { color: sqnColor(c.sqn), fontWeight: 700 })}
                {td((c.kelly * 100).toFixed(1) + '%')}
                {td((c.tp_rate * 100).toFixed(0) + '%', { color: '#22c55e' })}
                {td((c.sl_rate * 100).toFixed(0) + '%', { color: '#ef4444' })}
                {td((c.hz_rate * 100).toFixed(0) + '%', { color: '#8890a4' })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
