import { useEffect, useState } from 'react';
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
  const [fanHorizon, setFanHorizon]     = useState(null);
  const [enabledQuantiles, setEnabledQuantiles] = useState(() => new Set(['p50']));
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
          {signals.map(({ symbol, signal }) => (
            <div key={symbol} style={{
              padding: '5px 12px', borderRadius: 5, fontFamily: FONT, fontSize: 11,
              border: signal ? `1px solid ${signal.dir === 'long' ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)'}` : '1px solid #1e2230',
              background: signal ? (signal.dir === 'long' ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)') : '#111318',
              color: signal ? (signal.dir === 'long' ? '#22c55e' : '#ef4444') : '#3a3f52',
            }}>
              <span style={{ fontWeight: 700 }}>{symbol.replace('USDT', '')}</span>
              {signal
                ? <span style={{ marginLeft: 8 }}>{signal.dir.toUpperCase()} · {signal.horizon}h · {signal.threshold}% · {((signal.prob || 0) * 100).toFixed(0)}%</span>
                : <span style={{ marginLeft: 8, color: '#2a2f3d' }}>NO SIGNAL</span>
              }
            </div>
          ))}
          {!liveData && (
            <div style={{ padding: '5px 12px', borderRadius: 5, border: '1px solid #1e2230', color: '#505870', fontFamily: FONT, fontSize: 11 }}>
              No live data — run: python -m python.live_predict --loop
            </div>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: -8, flexWrap: 'wrap' }}>
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
        {selected && (
          <OHLCVChart symbol={selected} interval="1h" height={700}
                      fanData={fanData} showVolume showIntervalPicker maxCandles={10000} />
        )}

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
