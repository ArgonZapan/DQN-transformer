import { useCallback, useEffect, useRef, useState } from 'react';
import OHLCVChart from './OHLCVChart.jsx';
import { BarChart } from './charts.jsx';

const FONT = 'IBM Plex Mono,monospace';

// Keys derived from config.quantiles (e.g. 0.10 → "p10").
const qKey = (q) => `p${Math.round(q * 100)}`;

function ProbabilityChart({ horizonData }) {
  const longProbs  = horizonData?.long?.threshold_probs;
  const shortProbs = horizonData?.short?.threshold_probs;
  if (!longProbs || !shortProbs) return null;
  const keys = Object.keys(longProbs);
  const longVals  = keys.map(k => +((longProbs[k]  || 0) * 100).toFixed(1));
  const shortVals = keys.map(k => +((shortProbs[k] || 0) * 100).toFixed(1));
  return (
    <BarChart
      width={600} height={180}
      labels={keys.map(k => k.replace('pct', '%'))}
      yMin={0} yMax={100}
      groups={[
        { label: 'Long',  color: 'rgba(34,197,94,0.5)', border: '#22c55e', values: longVals  },
        { label: 'Short', color: 'rgba(239,68,68,0.5)', border: '#ef4444', values: shortVals },
      ]}
    />
  );
}

function QuantileFanSVG({ prediction, qKeys, width = 260, height = 90 }) {
  if (!prediction || qKeys.length < 3) return null;
  const horizons = prediction.horizons;
  const pad = { l: 30, r: 10, t: 8, b: 18 };
  const W = width - pad.l - pad.r;
  const H = height - pad.t - pad.b;

  // Outer band = first/last quantiles, median = middle key (or p50 fallback).
  const qLo  = qKeys[0];
  const qMid = qKeys.find(k => k === 'p50') ?? qKeys[Math.floor(qKeys.length / 2)];
  const qHi  = qKeys[qKeys.length - 1];

  const allVals = horizons.flatMap(h => [
    h.long?.quantiles?.[qHi]  || 0,
    -(h.short?.quantiles?.[qHi] || 0),
    0,
  ]);
  const yMin = Math.min(...allVals) * 1.1;
  const yMax = Math.max(...allVals) * 1.1;
  const yRange = (yMax - yMin) || 1;

  const horizonTimes = horizons.map(h => h.horizon_h || 0);
  const maxHorizonH = Math.max(...horizonTimes);

  const xOfTime = (h) => pad.l + (h / Math.max(1, maxHorizonH)) * W;
  const yOf = v => pad.t + H - ((v - yMin) / yRange) * H;

  const makePath = (vals) => vals.length
    ? vals.map((v, i) => `${i === 0 ? 'M' : 'L'} ${xOfTime(horizonTimes[i])},${yOf(v)}`).join(' ')
    : '';

  const longQ  = (q) => horizons.map(h => h.long?.quantiles?.[q]  || 0);
  const shortQ = (q) => horizons.map(h => h.short?.quantiles?.[q] || 0);
  const zero = yOf(0);
  const closeR = `L${xOfTime(maxHorizonH)},${zero} L${xOfTime(0)},${zero} Z`;

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <line x1={pad.l} x2={pad.l + W} y1={zero} y2={zero} stroke="#1e2230" strokeWidth={1} />
      <path d={`${makePath(longQ(qHi))} ${closeR}`} fill="rgba(34,197,94,0.07)" />
      <path d={`${makePath(longQ(qLo))} ${closeR}`} fill="rgba(34,197,94,0.18)" />
      <path d={makePath(longQ(qMid))} fill="none" stroke="rgba(34,197,94,0.85)" strokeWidth={1.5} />
      <path d={`${makePath(shortQ(qHi).map(v => -v))} ${closeR}`} fill="rgba(239,68,68,0.07)" />
      <path d={`${makePath(shortQ(qLo).map(v => -v))} ${closeR}`} fill="rgba(239,68,68,0.18)" />
      <path d={makePath(shortQ(qMid).map(v => -v))} fill="none" stroke="rgba(239,68,68,0.85)" strokeWidth={1.5} />
      {horizons.map((h, i) => (
        <text key={h.horizon_h} x={xOfTime(horizonTimes[i])} y={height - 2}
              textAnchor="middle" fontSize={8} fill="#3a3f52" fontFamily={FONT}>{h.horizon_h}h</text>
      ))}
      <text x={pad.l - 2} y={pad.t + 4} textAnchor="end" fontSize={7} fill="#3a3f52" fontFamily={FONT}>+{yMax.toFixed(1)}%</text>
      <text x={pad.l - 2} y={height - pad.b + 2} textAnchor="end" fontSize={7} fill="#3a3f52" fontFamily={FONT}>{yMin.toFixed(1)}%</text>
    </svg>
  );
}

// Long/short bias as long.qX / short.qX ratio over time, drawn in log space so
// the symmetry around 1.0 is preserved (ratio 2 and 0.5 are equidistant from 1).
// X axis = time (last BIAS_WINDOW_ANCHORS historical anchors + current);
// each line = one enabled quantile at the selected horizon.
const BIAS_WINDOW_ANCHORS = 60; // show ~20 days at 8h step instead of the full 60-day history

function BiasChart({ pred, qKeys, enabledQuantiles, threshold, horizonH, width = 720, height = 260 }) {
  if (!pred) return null;
  const active = qKeys.filter(q => enabledQuantiles.has(q));
  if (!active.length) {
    return <div style={{ fontFamily: FONT, fontSize: 10, color: '#505870' }}>Enable at least one quantile above to plot bias.</div>;
  }

  const findH = (hzList) => hzList?.find(h => h.horizon_h === horizonH) ?? hzList?.[0];

  // Use only the most recent BIAS_WINDOW_ANCHORS historical anchors so the chart
  // stays readable. historical[0] is the newest anchor, so we slice from the front.
  const recentHist = (pred.historical || []).slice(0, BIAS_WINDOW_ANCHORS);
  const samples = [];
  for (const hist of recentHist) {
    const hz = findH(hist.horizons);
    const t  = Date.parse(hist.anchor_time);
    if (hz && Number.isFinite(t)) samples.push({ t, hz });
  }
  const curHz = findH(pred.horizons);
  const curT  = Date.parse(pred.current_time);
  if (curHz && Number.isFinite(curT)) samples.push({ t: curT, hz: curHz });
  samples.sort((a, b) => a.t - b.t);

  if (samples.length < 2) {
    return <div style={{ fontFamily: FONT, fontSize: 10, color: '#505870' }}>
      Need at least 2 anchors — only {samples.length} available
      (check `hist_anchor_step_h` and warmup period).
    </div>;
  }

  const series = active.map(q => ({
    key: q,
    points: samples.map(({ t, hz }) => {
      const L = hz.long?.quantiles?.[q];
      const S = hz.short?.quantiles?.[q];
      if (!Number.isFinite(L) || !Number.isFinite(S) || L <= 0 || S <= 0) return null;
      return { t, r: L / S };
    }).filter(Boolean),
  })).filter(s => s.points.length);

  if (!series.length) return null;

  const pad = { l: 54, r: 14, t: 22, b: 32 };
  const W = width - pad.l - pad.r;
  const H = height - pad.t - pad.b;

  const allR   = series.flatMap(s => s.points.map(p => p.r));
  const logT   = Math.log(Math.max(1.0001, threshold));
  // Expand Y range to at least ±threshold so threshold lines are always visible
  // AND the actual data range, so lines never clip to a horizontal band.
  const dataMinLog = Math.min(...allR.map(Math.log));
  const dataMaxLog = Math.max(...allR.map(Math.log));
  const minLog  = Math.min(dataMinLog, -logT) - Math.abs(dataMinLog - dataMaxLog) * 0.1;
  const maxLog  = Math.max(dataMaxLog,  logT) + Math.abs(dataMinLog - dataMaxLog) * 0.1;
  const range   = (maxLog - minLog) || 1;

  const tMin = samples[0].t;
  const tMax = samples[samples.length - 1].t;
  const tRange = Math.max(1, tMax - tMin);
  const xOf = t => pad.l + ((t - tMin) / tRange) * W;
  const yOf = r => pad.t + H - ((Math.log(r) - minLog) / range) * H;

  const palette = ['#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#06b6d4', '#ef4444', '#eab308'];

  const yLong  = yOf(threshold);
  const yShort = yOf(1 / threshold);
  const yMid   = yOf(1);

  const fmtTime = (ms) => {
    const d = new Date(ms);
    return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
  };

  // Crossing stats for the status line
  const nLong  = series.flatMap(s => s.points).filter(p => p.r >= threshold).length;
  const nShort = series.flatMap(s => s.points).filter(p => p.r <= 1 / threshold).length;
  const lastR  = series[0]?.points.at(-1)?.r;

  // 3 intermediate X-axis ticks
  const xTicks = [0.25, 0.5, 0.75].map(frac => tMin + frac * tRange);

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ width: '100%', height }}>
        {/* Bias zone fills */}
        <rect x={pad.l} y={pad.t} width={W} height={Math.max(0, yLong - pad.t)} fill="rgba(34,197,94,0.07)" />
        <rect x={pad.l} y={yShort} width={W} height={Math.max(0, pad.t + H - yShort)} fill="rgba(239,68,68,0.07)" />

        {/* Reference lines */}
        <line x1={pad.l} x2={pad.l + W} y1={yMid}   y2={yMid}   stroke="#2a2f3d" strokeWidth={1} />
        <line x1={pad.l} x2={pad.l + W} y1={yLong}  y2={yLong}  stroke="rgba(34,197,94,0.6)" strokeWidth={1} strokeDasharray="4 3" />
        <line x1={pad.l} x2={pad.l + W} y1={yShort} y2={yShort} stroke="rgba(239,68,68,0.6)" strokeWidth={1} strokeDasharray="4 3" />

        {/* Y-axis labels */}
        <text x={pad.l - 4} y={yMid + 3}   textAnchor="end" fontSize={9} fill="#505870" fontFamily={FONT}>1.00×</text>
        <text x={pad.l - 4} y={Math.max(pad.t + 8, yLong + 3)}  textAnchor="end" fontSize={9} fill="rgba(34,197,94,0.85)" fontFamily={FONT}>{threshold.toFixed(2)}×</text>
        <text x={pad.l - 4} y={Math.min(pad.t + H - 2, yShort + 3)} textAnchor="end" fontSize={9} fill="rgba(239,68,68,0.85)" fontFamily={FONT}>{(1 / threshold).toFixed(2)}×</text>

        {/* X-axis ticks: start, 3 intermediate, end */}
        <text x={pad.l}           y={height - 8} textAnchor="start" fontSize={8} fill="#3a3f52" fontFamily={FONT}>{fmtTime(tMin)}</text>
        {xTicks.map((t, i) => (
          <text key={i} x={xOf(t)} y={height - 8} textAnchor="middle" fontSize={8} fill="#2a2f3d" fontFamily={FONT}>{fmtTime(t)}</text>
        ))}
        <text x={pad.l + W}       y={height - 8} textAnchor="end"   fontSize={8} fill="#3a3f52" fontFamily={FONT}>{fmtTime(tMax)}</text>

        {/* Tick marks */}
        {xTicks.map((t, i) => (
          <line key={i} x1={xOf(t)} x2={xOf(t)} y1={pad.t + H} y2={pad.t + H + 4} stroke="#2a2f3d" strokeWidth={1} />
        ))}

        {/* Data lines + triggered dots */}
        {series.map((s, si) => {
          const color = palette[si % palette.length];
          const path  = s.points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xOf(p.t).toFixed(1)},${yOf(p.r).toFixed(1)}`).join(' ');
          return (
            <g key={s.key}>
              <path d={path} fill="none" stroke={color} strokeWidth={1.5} opacity={0.85} />
              {s.points.map((p, i) => {
                const isLong  = p.r >= threshold;
                const isShort = p.r <= 1 / threshold;
                if (!isLong && !isShort) return null;
                return (
                  <circle key={i} cx={xOf(p.t).toFixed(1)} cy={yOf(p.r).toFixed(1)} r={3.5}
                    fill={isLong ? '#22c55e' : '#ef4444'}
                    stroke="#0b0c0f" strokeWidth={1} />
                );
              })}
            </g>
          );
        })}

        {/* Legend */}
        <g transform={`translate(${pad.l + 6}, ${pad.t - 14})`}>
          {series.map((s, si) => (
            <g key={s.key} transform={`translate(${si * 58}, 0)`}>
              <rect width={8} height={8} rx={1} fill={palette[si % palette.length]} />
              <text x={12} y={8} fontSize={10} fill="#8890a4" fontFamily={FONT}>{s.key}</text>
            </g>
          ))}
          <text x={series.length * 58 + 8} y={8} fontSize={10} fill="#3a3f52" fontFamily={FONT}>@ {horizonH}h</text>
        </g>
      </svg>

      {/* Status line below chart */}
      <div style={{ display: 'flex', gap: 16, marginTop: 6, paddingLeft: 4, flexWrap: 'wrap' }}>
        {lastR != null && (
          <span style={{ fontFamily: FONT, fontSize: 10, color: lastR >= threshold ? '#22c55e' : lastR <= 1 / threshold ? '#ef4444' : '#505870' }}>
            current ratio: {lastR.toFixed(3)}×
          </span>
        )}
        <span style={{ fontFamily: FONT, fontSize: 10, color: nLong > 0 ? '#22c55e' : '#3a3f52' }}>
          ▲ long bias: {nLong} pts
        </span>
        <span style={{ fontFamily: FONT, fontSize: 10, color: nShort > 0 ? '#ef4444' : '#3a3f52' }}>
          ▼ short bias: {nShort} pts
        </span>
        <span style={{ fontFamily: FONT, fontSize: 10, color: '#2a2f3d' }}>
          window: {samples.length} anchors
        </span>
      </div>
    </div>
  );
}

function detectSignal(prediction, bestCombo) {
  if (!prediction || !bestCombo) return null;
  const { horizon, threshold, entry_prob, direction } = bestCombo;
  const horizonData = prediction.horizons.find(h => h.horizon_h === horizon);
  if (!horizonData) return null;
  const key = `${threshold}pct`;
  const dirs = direction === 'both' ? ['long', 'short'] : [direction];
  for (const dir of dirs) {
    const prob = horizonData[dir]?.threshold_probs?.[key];
    if (prob != null && prob >= entry_prob) return { dir, prob, horizon, threshold };
  }
  return null;
}

function SymbolCard({ pred, isSelected, onClick, signal, qKeys }) {
  const fmt = (p) => p > 100 ? p.toLocaleString('en-US', { maximumFractionDigits: 2 }) : p.toFixed(4);
  const h24 = pred.horizons.find(h => h.horizon_h === 24) ?? pred.horizons[0];
  const midKey = qKeys.find(k => k === 'p50') ?? qKeys[Math.floor(qKeys.length / 2)];
  const p50L = h24?.long?.quantiles?.[midKey];
  const p50S = h24?.short?.quantiles?.[midKey];

  return (
    <div onClick={onClick} style={{
      padding: '10px 12px', cursor: 'pointer', borderBottom: '1px solid #181b26',
      background: isSelected ? '#131a2e' : 'transparent',
      borderLeft: isSelected ? '2px solid #3b82f6' : '2px solid transparent',
    }}
      onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = '#0f1118'; }}
      onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = 'transparent'; }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontFamily: FONT, fontWeight: 700, color: '#e2e4ea', fontSize: 12 }}>
          {pred.symbol.replace('USDT', '')}<span style={{ color: '#3a3f52', fontWeight: 400 }}>/USDT</span>
        </span>
        {signal && (
          <span style={{
            fontSize: 9, fontFamily: FONT, fontWeight: 700,
            color: signal.dir === 'long' ? '#22c55e' : '#ef4444',
            background: signal.dir === 'long' ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
            border: `1px solid ${signal.dir === 'long' ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'}`,
            borderRadius: 3, padding: '1px 5px',
          }}>{signal.dir.toUpperCase()}</span>
        )}
      </div>
      <div style={{ fontFamily: FONT, color: '#c5c9d6', fontSize: 13, marginBottom: 6 }}>
        {fmt(pred.current_price)}
      </div>
      <QuantileFanSVG prediction={pred} qKeys={qKeys} width={188} height={60} />
      {p50L != null && (
        <div style={{ display: 'flex', gap: 10, marginTop: 4 }}>
          <span style={{ fontSize: 9, color: '#22c55e', fontFamily: FONT }}>&#9650; {p50L}% 24h</span>
          <span style={{ fontSize: 9, color: '#ef4444', fontFamily: FONT }}>&#9660; {p50S}% 24h</span>
        </div>
      )}
    </div>
  );
}

// Reconnects with exponential backoff up to 10s.
function useLiveStream() {
  const [data, setData]     = useState(null);
  const [status, setStatus] = useState('connecting');

  useEffect(() => {
    let es;
    let retryMs = 300;
    let timeout;
    let cancelled = false;

    const connect = () => {
      es = new EventSource('/api/live');
      es.onopen = () => {
        retryMs = 300;
        setStatus('connected');
      };
      es.onmessage = (e) => {
        try {
          const parsed = JSON.parse(e.data);
          setData(parsed);
          setStatus('connected');
        } catch { /* ignore malformed */ }
      };
      es.onerror = () => {
        setStatus('error');
        es.close();
        if (cancelled) return;
        timeout = setTimeout(connect, retryMs);
        retryMs = Math.min(retryMs * 2, 10_000);
      };
    };

    connect();
    return () => { cancelled = true; clearTimeout(timeout); es?.close(); };
  }, []);

  return { data, status };
}

function useAppConfig() {
  const [config, setConfig] = useState(null);
  useEffect(() => {
    let cancelled = false;
    fetch('/api/config')
      .then(r => r.json())
      .then(c => { if (!cancelled) setConfig(c); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);
  return config;
}

export default function LiveSection({ bestCombo }) {
  const config = useAppConfig();
  const { data: liveData, status: sseStatus } = useLiveStream();

  const symbols   = config?.symbols ?? [];
  const horizonsH = config?.horizons_hours ?? [];
  const qKeys     = (config?.quantiles ?? []).map(qKey);

  const [selected, setSelected] = useState(null);
  const [selectedHorizon, setSelectedHorizon] = useState(null);
  const [fanHorizon, setFanHorizon]     = useState(72);
  const [enabledQuantiles, setEnabledQuantiles] = useState(() => new Set(['p75', 'p90']));
  const [rrCombo, setRrCombo] = useState(null);  // { symbol, dir, horizon, threshold }
  const [combosSortKey, setCombosSortKey] = useState('edge');
  const [combosSortDir, setCombosSortDir] = useState('desc');
  const [biasEnabled, setBiasEnabled]     = useState(false);
  // 1.05 = 5% bias threshold — calibrated to actual L/S ratio range (~0.88–1.11)
  const [biasThreshold, setBiasThreshold] = useState(1.05);

  // ── Continuous prediction mode ────────────────────────────────────────────
  const [viewMode, setViewMode]                     = useState('fan');
  const [continuousPredictions, setContPredictions] = useState(null);
  const [contPredLoading, setContPredLoading]       = useState(false);
  const [currentTf, setCurrentTf]                   = useState('15m');
  const rangeChangeTimer                            = useRef(null);

  const fetchContPredictions = useCallback(async (sym, from_ms, to_ms, h, tf) => {
    setContPredLoading(true);
    try {
      const r = await fetch(
        `/api/live/predict-range?symbol=${sym}&from_ms=${from_ms}&to_ms=${to_ms}&horizon_h=${h}&tf=${tf}`
      );
      if (r.ok) setContPredictions(await r.json());
    } catch { /* silently ignore network errors */ }
    finally { setContPredLoading(false); }
  }, []);

  const handleVisibleRangeChange = useCallback(({ from_ms, to_ms }) => {
    if (viewMode !== 'continuous' || !selected) return;
    clearTimeout(rangeChangeTimer.current);
    rangeChangeTimer.current = setTimeout(() => {
      fetchContPredictions(selected, from_ms, to_ms, fanHorizon ?? horizonsH[0] ?? 8, currentTf);
    }, 600);
  }, [viewMode, selected, fanHorizon, horizonsH, currentTf, fetchContPredictions]);

  // Clear continuous predictions when switching back to fan or changing symbol.
  useEffect(() => {
    if (viewMode !== 'continuous') setContPredictions(null);
  }, [viewMode]);
  useEffect(() => { setContPredictions(null); }, [selected]);
  const toggleQuantile = (q) => setEnabledQuantiles(prev => {
    const next = new Set(prev);
    if (next.has(q)) next.delete(q); else next.add(q);
    return next;
  });

  // Initialise selection once config arrives.
  useEffect(() => {
    if (selected == null && symbols.length)        setSelected(symbols[0]);
    if (selectedHorizon == null && horizonsH.length) {
      setSelectedHorizon(horizonsH.includes(24) ? 24 : horizonsH[0]);
    }
  }, [symbols, horizonsH, selected, selectedHorizon]);

  const lastRefresh = liveData?.generated_at ? new Date(liveData.generated_at) : null;

  const selPred = liveData?.predictions?.find(p => p.symbol === selected);
  const selHorizonData = selPred?.horizons?.find(h => h.horizon_h === selectedHorizon);

  const allHorizonHours = selPred?.horizons?.map(h => h.horizon_h) ?? horizonsH;
  const maxHorizonH     = Math.max(1, ...allHorizonHours);
  // Server is the only source of truth for stride.
  const stepH  = selPred?.hist_anchor_step_h ?? Math.min(...(allHorizonHours.length ? allHorizonHours : [1]));
  const stride = Math.max(1, fanHorizon ? Math.round(fanHorizon / stepH) : Math.round(maxHorizonH / stepH));

  const fanData = selPred ? {
    currentPrice:    selPred.current_price,
    currentTime:     selPred.current_time ? Math.floor(Date.parse(selPred.current_time) / 1000) : null,
    selectedHorizon: fanHorizon,
    enabledQuantiles: Array.from(enabledQuantiles),
    qKeys,
    horizons: selPred.horizons.map(h => ({
      h: h.horizon_h,
      long:  h.long?.quantiles  || {},
      short: h.short?.quantiles || {},
    })),
    historicalFans: (selPred.historical || [])
      .filter((_, i) => i % stride === 0)
      .map(hist => ({
        anchorTime:  Math.floor(Date.parse(hist.anchor_time) / 1000),
        anchorPrice: hist.anchor_price,
        horizons: hist.horizons.map(h => ({
          h: h.horizon_h,
          long:  h.long?.quantiles  || {},
          short: h.short?.quantiles || {},
        })),
      })),
  } : null;

  const signals = (liveData?.predictions || []).map(p => ({
    symbol: p.symbol, signal: detectSignal(p, bestCombo),
  }));

  // Option 4: scan all (horizon, dir, qTP, qSL) combos using quantile predictions.
  // Quantiles are independent distributions for long/short tails so p_TP and p_SL
  // don't double-count joint touches (unlike threshold_probs).
  // pXX quantile → P(magnitude ≥ this value) = 1 − XX/100.
  const MIN_P_TP   = 0.10;
  const MIN_SL_PCT = 0.5;   // ignore stops tighter than 0.5% (degenerate huge R)
  const SKIP_QUANTILES = new Set(['p10']);  // tail magnitudes too small / unreliable
  const parseQ = (k) => {
    const n = parseInt(k.replace(/^p/, ''), 10);
    return isFinite(n) ? n / 100 : null;
  };
  // Returns ALL valid combos (with first-touch correction). Caller filters/ranks.
  const computeCombos = (pred) => {
    if (!pred) return [];
    const rows = [];
    for (const h of pred.horizons || []) {
      const longQ  = h.long?.quantiles  || {};
      const shortQ = h.short?.quantiles || {};
      const tryDir = (dir, winQ, lossQ) => {
        for (const tpKey of Object.keys(winQ)) {
          if (SKIP_QUANTILES.has(tpKey)) continue;
          const qTP = parseQ(tpKey);
          if (qTP == null) continue;
          const pTP = 1 - qTP;
          if (pTP < MIN_P_TP) continue;
          const TP = winQ[tpKey];
          if (!isFinite(TP) || TP <= 0) continue;
          for (const slKey of Object.keys(lossQ)) {
            if (SKIP_QUANTILES.has(slKey)) continue;
            const qSL = parseQ(slKey);
            if (qSL == null) continue;
            const pSL = 1 - qSL;
            const SL = lossQ[slKey];
            if (!isFinite(SL) || SL < MIN_SL_PCT) continue;
            if (pTP + pSL > 1) continue;
            const pTPc = pTP * (1 - pSL / 2);
            const pSLc = pSL * (1 - pTP / 2);
            const expReward = pTPc * TP;
            const expRisk   = pSLc * SL;
            if (expRisk <= 0) continue;
            const edge   = expReward / expRisk;
            const rr     = TP / SL;
            const pTotal = pTPc + pSLc;
            const pCond  = pTotal > 0 ? pTPc / pTotal : 0;
            const kelly  = rr > 0 ? (rr * pCond - (1 - pCond)) / rr : 0;
            rows.push({
              dir, horizon: h.horizon_h,
              tp: TP, sl: SL, qTP: tpKey, qSL: slKey,
              pTP, pSL, rr, edge, kelly,
            });
          }
        }
      };
      tryDir('long',  longQ,  shortQ);
      tryDir('short', shortQ, longQ);
    }
    return rows;
  };

  const bestERR = (liveData?.predictions || []).map(p => {
    const rows = computeCombos(p).filter(r => r.edge > 1);
    const best = rows.reduce((a, b) => (a == null || b.edge > a.edge ? b : a), null);
    return { symbol: p.symbol, best };
  });

  const statusColor = { connected: '#22c55e', connecting: '#f59e0b', error: '#ef4444' };
  const cardS  = { background: '#111318', border: '1px solid #1e2230', borderRadius: 8, padding: 14 };
  const titleS = { fontSize: 10, color: '#505870', fontFamily: FONT, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 };

  const QUANTILE_KEYS = ['p10', 'p25', 'p50', 'p75', 'p90'];

  return (
    <div style={{ display: 'flex', gap: 16, minHeight: 600 }}>
      <div style={{ width: 212, flexShrink: 0, background: '#111318', border: '1px solid #1e2230', borderRadius: 8, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '10px 12px', borderBottom: '1px solid #1e2230', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 10, color: '#505870', fontFamily: FONT, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Symbols</span>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: statusColor[sseStatus] || '#6b7280', display: 'inline-block' }} />
        </div>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {liveData ? liveData.predictions.map(p => (
            <SymbolCard key={p.symbol} pred={p} isSelected={p.symbol === selected}
              onClick={() => setSelected(p.symbol)}
              signal={signals.find(s => s.symbol === p.symbol)?.signal}
              qKeys={qKeys} />
          )) : symbols.map(s => (
            <div key={s} style={{ padding: '10px 12px', borderBottom: '1px solid #181b26', color: '#3a3f52', fontSize: 12, fontFamily: FONT }}>{s}</div>
          ))}
        </div>
        <div style={{ padding: '8px 12px', borderTop: '1px solid #1e2230', fontSize: 9, color: '#3a3f52', fontFamily: FONT }}>
          {sseStatus === 'connected' ? 'Live (SSE)' : sseStatus === 'error' ? 'SSE error — reconnecting…' : 'Connecting…'}
          {lastRefresh && <div style={{ marginTop: 2 }}>Updated: {lastRefresh.toLocaleTimeString()}</div>}
          {liveData?.model_checkpoint && <div style={{ marginTop: 2, color: '#2a2f3d', wordBreak: 'break-all' }}>{liveData.model_checkpoint}</div>}
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 14, overflow: 'auto' }}>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {bestERR.map(({ symbol, best }) => {
            const active = best && rrCombo?.symbol === symbol
              && rrCombo.dir === best.dir && rrCombo.horizon === best.horizon
              && rrCombo.tp === best.tp && rrCombo.sl === best.sl;
            const onClickBadge = () => {
              if (!best) return;
              setSelected(symbol);
              const pred = liveData?.predictions?.find(x => x.symbol === symbol);
              const entryPrice = pred?.current_price ?? null;
              setRrCombo(active ? null : {
                symbol, dir: best.dir, horizon: best.horizon,
                tp: best.tp, sl: best.sl, entryPrice,
              });
            };
            return (
              <div key={symbol}
                onClick={onClickBadge}
                style={{
                  padding: '5px 12px', borderRadius: 5, fontFamily: FONT, fontSize: 11,
                  cursor: best ? 'pointer' : 'default',
                  border: best
                    ? `1px solid ${best.dir === 'long' ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)'}`
                    : '1px solid #1e2230',
                  background: active
                    ? (best.dir === 'long' ? 'rgba(34,197,94,0.20)' : 'rgba(239,68,68,0.20)')
                    : best ? (best.dir === 'long' ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)') : '#111318',
                  color: best ? (best.dir === 'long' ? '#22c55e' : '#ef4444') : '#3a3f52',
                  boxShadow: active ? `0 0 0 1px ${best.dir === 'long' ? '#22c55e' : '#ef4444'}` : 'none',
                  transition: 'background 0.12s, box-shadow 0.12s',
                }}>
                <span style={{ fontWeight: 700 }}>{symbol.replace('USDT', '')}</span>
                {best
                  ? <span style={{ marginLeft: 8 }}>
                      {best.dir.toUpperCase()} · {best.horizon}h · TP{best.tp.toFixed(1)}({best.qTP})/SL{best.sl.toFixed(1)}({best.qSL}) · RR{best.rr.toFixed(2)} · Edge{best.edge.toFixed(2)}
                      <span style={{ marginLeft: 6, color: best.dir === 'long' ? 'rgba(34,197,94,0.55)' : 'rgba(239,68,68,0.55)' }}>
                        K{(best.kelly * 100).toFixed(0)}%
                      </span>
                    </span>
                  : <span style={{ marginLeft: 8, color: '#2a2f3d' }}>NO EDGE</span>
                }
              </div>
            );
          })}
          {!liveData && (
            <div style={{ padding: '5px 12px', borderRadius: 5, border: '1px solid #1e2230', color: '#505870', fontFamily: FONT, fontSize: 11 }}>
              No live data — run: python -m python.live_predict --loop
            </div>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: -8, flexWrap: 'wrap' }}>
          {/* View mode toggle */}
          {(['fan', 'continuous']).map(m => (
            <button key={m} onClick={() => setViewMode(m)} style={{
              padding: '2px 10px', fontFamily: FONT, fontSize: 10,
              background: viewMode === m ? 'rgba(139,92,246,0.18)' : 'transparent',
              color: viewMode === m ? '#a78bfa' : '#505870',
              border: '1px solid ' + (viewMode === m ? '#a78bfa' : '#1e2230'),
              borderRadius: 3, cursor: 'pointer',
            }}>{m === 'fan' ? 'Fan' : 'Continuous'}</button>
          ))}
          {contPredLoading && (
            <span style={{ fontFamily: FONT, fontSize: 9, color: '#a78bfa' }}>loading…</span>
          )}
          <span style={{ fontFamily: FONT, fontSize: 10, color: '#1e2230', marginLeft: 4 }}>│</span>
          <span style={{ fontFamily: FONT, fontSize: 10, color: '#505870', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Fan:</span>
          {[null, ...horizonsH].map(h => (
            <button key={h ?? 'all'} onClick={() => setFanHorizon(h)} style={{
              padding: '2px 8px', fontFamily: FONT, fontSize: 10,
              background: fanHorizon === h ? 'rgba(59,130,246,0.15)' : 'transparent',
              color: fanHorizon === h ? '#3b82f6' : '#505870',
              border: '1px solid ' + (fanHorizon === h ? '#3b82f6' : '#1e2230'),
              borderRadius: 3, cursor: 'pointer',
            }}>{h === null ? 'All' : `${h}h`}</button>
          ))}
          <span style={{ fontFamily: FONT, fontSize: 10, color: '#505870', textTransform: 'uppercase', letterSpacing: '0.08em', marginLeft: 12 }}>Quantiles:</span>
          {QUANTILE_KEYS.map(q => {
            const active = enabledQuantiles.has(q);
            return (
              <button key={q} onClick={() => toggleQuantile(q)} style={{
                padding: '2px 8px', fontFamily: FONT, fontSize: 10,
                background: active ? 'rgba(59,130,246,0.15)' : 'transparent',
                color: active ? '#3b82f6' : '#505870',
                border: '1px solid ' + (active ? '#3b82f6' : '#1e2230'),
                borderRadius: 3, cursor: 'pointer',
              }}>{q}</button>
            );
          })}
        </div>
        {selected && (() => {
          const rrLines = (rrCombo && rrCombo.symbol === selected && rrCombo.entryPrice)
            ? (() => {
                const ep = rrCombo.entryPrice;
                const tpPct = rrCombo.tp / 100;
                const slPct = rrCombo.sl / 100;
                const isLong = rrCombo.dir === 'long';
                const tp = ep * (1 + (isLong ? tpPct : -tpPct));
                const sl = ep * (1 + (isLong ? -slPct : slPct));
                return [
                  { price: ep, color: '#e2e4ea', title: 'EP' },
                  { price: tp, color: '#22c55e', title: `TP ${isLong ? '+' : '-'}${rrCombo.tp.toFixed(1)}%` },
                  { price: sl, color: '#ef4444', title: `SL ${isLong ? '-' : '+'}${rrCombo.sl.toFixed(1)}%` },
                ];
              })()
            : [];
          return (
            <OHLCVChart symbol={selected} interval="15m" height={700}
                        fanData={viewMode === 'fan' ? fanData : null}
                        priceLines={rrLines}
                        showVolume showIntervalPicker maxCandles={10000}
                        viewMode={viewMode}
                        continuousPredictions={continuousPredictions}
                        enabledQuantiles={Array.from(enabledQuantiles)}
                        onVisibleRangeChange={handleVisibleRangeChange}
                        onTfChange={setCurrentTf} />
          );
        })()}

        <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 14 }}>
          <div style={{ ...cardS, minWidth: 120 }}>
            <div style={titleS}>Horizon</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {horizonsH.map(h => (
                <button key={h} onClick={() => setSelectedHorizon(h)} style={{
                  padding: '6px 14px', fontFamily: FONT, fontSize: 11,
                  background: selectedHorizon === h ? 'rgba(59,130,246,0.15)' : 'transparent',
                  color: selectedHorizon === h ? '#3b82f6' : '#505870',
                  border: '1px solid ' + (selectedHorizon === h ? '#3b82f6' : '#1e2230'),
                  borderRadius: 4, cursor: 'pointer', textAlign: 'left',
                }}>{h}h</button>
              ))}
            </div>
          </div>

          <div style={cardS}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
              <span style={titleS}>Threshold Probabilities — {selected?.replace('USDT', '')} {selectedHorizon}h</span>
              {selHorizonData && (
                <div style={{ display: 'flex', gap: 16 }}>
                  {['long', 'short'].map(dir => {
                    const midKey = qKeys.find(k => k === 'p50') ?? qKeys[Math.floor(qKeys.length / 2)];
                    const p50 = selHorizonData?.[dir]?.quantiles?.[midKey];
                    return (
                      <span key={dir} style={{ fontSize: 10, fontFamily: FONT, color: dir === 'long' ? '#22c55e' : '#ef4444' }}>
                        {dir} {midKey}: {p50}%
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
            <ProbabilityChart horizonData={selHorizonData} />
          </div>
        </div>

        <div style={cardS}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10, flexWrap: 'wrap', gap: 8 }}>
            <span style={titleS}>Long/Short Bias — {selected?.replace('USDT', '')}</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontFamily: FONT, fontSize: 10, color: '#8890a4', cursor: 'pointer' }}>
                <input type="checkbox" checked={biasEnabled} onChange={e => setBiasEnabled(e.target.checked)} />
                Enabled
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontFamily: FONT, fontSize: 10, color: biasEnabled ? '#8890a4' : '#3a3f52' }}>
                Threshold ×
                <input type="number" min={1.01} step={0.01} value={biasThreshold}
                  disabled={!biasEnabled}
                  onChange={e => {
                    const v = parseFloat(e.target.value);
                    if (Number.isFinite(v) && v > 1) setBiasThreshold(v);
                  }}
                  style={{
                    width: 60, padding: '2px 6px', fontFamily: FONT, fontSize: 11,
                    background: '#0f1014', color: '#c5c9d6',
                    border: '1px solid #1e2230', borderRadius: 3,
                  }} />
              </label>
            </div>
          </div>
          {biasEnabled
            ? (selPred
                ? <BiasChart pred={selPred} qKeys={qKeys}
                    enabledQuantiles={enabledQuantiles} threshold={biasThreshold}
                    horizonH={selectedHorizon ?? selPred.horizons[0]?.horizon_h} />
                : <div style={{ fontFamily: FONT, fontSize: 10, color: '#505870' }}>No prediction data.</div>)
            : <div style={{ fontFamily: FONT, fontSize: 10, color: '#3a3f52' }}>Disabled.</div>}
        </div>

        {selPred && (
          <div style={cardS}>
            <div style={titleS}>Quantile Predictions — {selected?.replace('USDT', '')}</div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', width: '100%' }}>
                <thead>
                  <tr style={{ background: '#0f1014' }}>
                    {['Horizon', 'Dir', ...qKeys].map(h => (
                      <th key={h} style={{ padding: '5px 10px', textAlign: 'right', fontSize: 10, color: '#505870', fontFamily: FONT, fontWeight: 400, borderBottom: '1px solid #1e2230', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {selPred.horizons.flatMap(h => ['long', 'short'].map(dir => (
                    <tr key={`${h.horizon_h}-${dir}`} style={{ borderBottom: '1px solid #181b26' }}>
                      <td style={{ padding: '4px 10px', textAlign: 'right', fontSize: 11, color: '#8890a4', fontFamily: FONT }}>{h.horizon_h}h</td>
                      <td style={{ padding: '4px 10px', textAlign: 'right', fontSize: 11, fontFamily: FONT, color: dir === 'long' ? '#22c55e' : '#ef4444' }}>{dir}</td>
                      {qKeys.map(q => (
                        <td key={q} style={{ padding: '4px 10px', textAlign: 'right', fontSize: 11, fontFamily: FONT, color: '#c5c9d6' }}>
                          {dir === 'long' ? '+' : '-'}{h[dir]?.quantiles?.[q] ?? '—'}%
                        </td>
                      ))}
                    </tr>
                  )))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
