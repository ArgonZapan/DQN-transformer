import { useEffect, useMemo, useRef, useState } from 'react';
import OHLCVChart from './OHLCVChart.jsx';
import { LineChart, Histogram, Donut } from './charts.jsx';

const FONT = 'IBM Plex Mono,monospace';

function pickTfForHorizon(h) {
  if (h <= 12) return '1h';
  if (h <= 48) return '4h';
  return '4h';
}

function dynamicBins(n) {
  if (n < 5) return Math.max(2, n);
  return Math.max(5, Math.min(20, Math.ceil(Math.sqrt(n))));
}

function downloadCsv(rows, cols, filename) {
  if (!rows.length) return;
  const lines = [cols.join(',')];
  rows.forEach(r => lines.push(cols.map(c => JSON.stringify(r[c] ?? '')).join(',')));
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function EquityCurveChart({ trades }) {
  const points = useMemo(() => {
    if (!trades?.length) return [];
    let eq = 1;
    // Backend stores t.return as a fraction (e.g. 0.0594 = 5.94%), not percent.
    return trades.map(t => {
      eq *= (1 + (t.return || 0));
      return { x: new Date(t.entry_time).toLocaleDateString(), y: +((eq - 1) * 100).toFixed(2) };
    });
  }, [trades]);
  if (!points.length) return null;
  const final = points[points.length - 1].y;
  const color = final >= 0 ? '#22c55e' : '#ef4444';
  const fill = final >= 0 ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)';
  return <LineChart width={1200} height={220} points={points} color={color} fill={fill} />;
}

function ReturnHistogram({ trades }) {
  if (!trades?.length) return null;
  // t.return is a fraction; convert to percent for display.
  return <Histogram width={600} height={220} values={trades.map(t => t.return * 100)} bins={dynamicBins(trades.length)} logY />;
}

function ExitDonut({ combo }) {
  if (!combo) return null;
  const trades = combo.trades || [];
  let tp = 0, sl = 0, hz = 0;
  if (trades.length) {
    trades.forEach(t => {
      if (t.exit_type === 'tp') tp++;
      else if (t.exit_type === 'sl') sl++;
      else hz++;
    });
    const total = tp + sl + hz || 1;
    tp = Math.round((tp / total) * 100);
    sl = Math.round((sl / total) * 100);
    hz = Math.max(0, 100 - tp - sl);
  } else {
    tp = Math.round((combo.tp_rate || 0) * 100);
    sl = Math.round((combo.sl_rate || 0) * 100);
    hz = Math.max(0, 100 - tp - sl);
  }
  return <Donut width={260} height={170} slices={[
    { label: 'TP', value: tp, color: '#22c55e' },
    { label: 'SL', value: sl, color: '#ef4444' },
    { label: 'HZ', value: hz, color: '#6b7280' },
  ]} />;
}

function buildTradeMarkers(combo) {
  const trades = combo?.trades || [];
  const out = [];
  trades.forEach(t => {
    const isLong = (t.direction || combo.direction) === 'long';
    if (t.entry_time) {
      out.push({
        time_ms: t.entry_time,
        position: isLong ? 'belowBar' : 'aboveBar',
        color: isLong ? '#22c55e' : '#ef4444',
        shape: isLong ? 'arrowUp' : 'arrowDown',
        text: isLong ? 'L' : 'S',
      });
    }
    if (t.exit_time) {
      const tp = t.exit_type === 'tp';
      const sl = t.exit_type === 'sl';
      out.push({
        time_ms: t.exit_time,
        position: isLong ? 'aboveBar' : 'belowBar',
        color: tp ? '#22c55e' : sl ? '#ef4444' : '#9ca3af',
        shape: 'circle',
        text: tp ? 'TP' : sl ? 'SL' : 'HZ',
      });
    }
  });
  return out;
}

function DetailPanel({ combo, onClose }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  if (!combo) return null;
  const tradeMarkers = buildTradeMarkers(combo);
  const tf = pickTfForHorizon(combo.horizon);
  const cardS = { background: '#111318', border: '1px solid #1e2230', borderRadius: 6, padding: 12, display: 'flex', flexDirection: 'column' };
  const titleS = { fontSize: 10, color: '#505870', fontFamily: FONT, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 };

  const exportTrades = () => downloadCsv(
    combo.trades || [],
    ['entry_time', 'exit_time', 'direction', 'exit_type', 'return'],
    `trades_${combo.symbol}_${combo.horizon}h_${combo.threshold}pct_${combo.direction}.csv`
  );

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 100, background: '#0b0c0f' }}
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div style={{ width: '100vw', height: '100vh', background: '#0b0c0f', padding: 20, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 14, boxSizing: 'border-box' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
          <span style={{ fontFamily: FONT, fontWeight: 700, color: '#e2e4ea', fontSize: 15 }}>
            {combo.symbol} · {combo.horizon}h · {combo.threshold}% · {combo.direction.toUpperCase()}
          </span>
          <span style={{ fontFamily: FONT, fontSize: 12, color: combo.sqn >= 2 ? '#22c55e' : combo.sqn >= 1 ? '#f59e0b' : '#ef4444' }}>
            SQN {combo.sqn}
          </span>
          <span style={{ fontFamily: FONT, fontSize: 12, color: '#505870' }}>
            {combo.n_trades} trades{combo.n_trades < 30 ? ' *' : ''}
          </span>
          <button onClick={exportTrades} style={{ marginLeft: 'auto', background: 'transparent', border: '1px solid #1e2230', color: '#8890a4', borderRadius: 4, padding: '4px 10px', cursor: 'pointer', fontFamily: FONT, fontSize: 11 }}>↓ Trades CSV</button>
          <button onClick={onClose} style={{ background: 'transparent', border: '1px solid #1e2230', color: '#8890a4', borderRadius: 4, padding: '4px 12px', cursor: 'pointer', fontFamily: FONT, fontSize: 11 }}>✕ Close (Esc)</button>
        </div>

        <div style={{ display: 'flex', gap: 2 }}>
          {[
            { label: 'P(Win)', value: ((combo.p_win || 0) * 100).toFixed(1) + '%' },
            { label: 'Avg +', value: '+' + Math.abs(combo.avg_win || 0).toFixed(2) + '%' },
            { label: 'Avg -', value: '-' + Math.abs(combo.avg_loss || 0).toFixed(2) + '%' },
            { label: 'EV', value: (combo.ev || 0) + '%' },
            { label: 'Sharpe', value: combo.sharpe || 0 },
            { label: 'Kelly', value: ((combo.kelly || 0) * 100).toFixed(1) + '%' },
            { label: 'Max DD', value: ((combo.max_dd || 0) * 100).toFixed(1) + '%' },
            { label: 'Total', value: (combo.total_return || 0) + '%' },
          ].map(({ label, value }) => (
            <div key={label} style={{ flex: 1, background: '#111318', border: '1px solid #1e2230', borderRadius: 5, padding: '6px 10px' }}>
              <div style={{ fontSize: 9, color: '#505870', fontFamily: FONT, textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label}</div>
              <div style={{ fontSize: 13, color: '#e2e4ea', fontFamily: FONT, fontWeight: 600, marginTop: 2 }}>{value}</div>
            </div>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: 'minmax(360px, 1fr) minmax(220px, auto) minmax(220px, auto)', gap: 12, flex: 1 }}>
          <div style={{ ...cardS, gridColumn: '1 / -1' }}>
            <div style={titleS}>OHLCV + Trade Markers ({tf})</div>
            <div style={{ flex: 1, minHeight: 0 }}>
              <OHLCVChart symbol={combo.symbol} interval={tf} height={420} markers={tradeMarkers} showVolume={false} showIntervalPicker maxCandles={1000} />
            </div>
          </div>
          <div style={{ ...cardS, gridColumn: '1 / -1' }}>
            <div style={titleS}>Equity Curve (compounded)</div>
            <div style={{ flex: 1 }}><EquityCurveChart trades={combo.trades || []} /></div>
          </div>
          <div style={cardS}>
            <div style={titleS}>Return Distribution</div>
            <div style={{ flex: 1 }}><ReturnHistogram trades={combo.trades || []} /></div>
          </div>
          <div style={cardS}>
            <div style={titleS}>Exit Type Breakdown</div>
            <div style={{ flex: 1 }}><ExitDonut combo={combo} /></div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function StrategySection({ onBestCombo }) {
  const [combos, setCombos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [meta, setMeta] = useState(null);
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState('');
  const [selected, setSelected] = useState(null);
  const [sortKey, setSortKey] = useState('sqn');
  const [sortDir, setSortDir] = useState(-1);
  const [filters, setFilters] = useState({ symbol: 'all', horizon: 'all', threshold: 'all', direction: 'all', minTrades: 0, minSqn: 0, minSharpe: -99 });

  const onBestRef = useRef(onBestCombo);
  useEffect(() => { onBestRef.current = onBestCombo; }, [onBestCombo]);

  useEffect(() => {
    fetch('/api/runs').then(r => r.json()).then(data => {
      setRuns(data);
      if (data.length > 0) setSelectedRun(data[data.length - 1]);
      else setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedRun) return;
    setLoading(true); setError(null);
    fetch(`/api/strategy/${selectedRun}`)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(data => {
        const all = data.combos || [];
        setCombos(all);
        setMeta(data);
        if (all.length > 0) {
          const best = [...all].sort((a, b) => (b.sqn || 0) - (a.sqn || 0))[0];
          onBestRef.current?.(best);
        }
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [selectedRun]);

  const opts = useMemo(() => {
    const symbols = [...new Set(combos.map(c => c.symbol))].sort();
    const horizons = [...new Set(combos.map(c => c.horizon))].sort((a, b) => a - b);
    const thresholds = [...new Set(combos.map(c => c.threshold))].sort((a, b) => a - b);
    const directions = [...new Set(combos.map(c => c.direction))].sort();
    return { symbols, horizons, thresholds, directions };
  }, [combos]);

  const handleSort = (key) => {
    if (sortKey === key) setSortDir(d => -d);
    else { setSortKey(key); setSortDir(-1); }
  };

  const filtered = useMemo(() => combos
    .filter(c =>
      (filters.symbol === 'all' || c.symbol === filters.symbol) &&
      (filters.horizon === 'all' || c.horizon === +filters.horizon) &&
      (filters.threshold === 'all' || c.threshold === +filters.threshold) &&
      (filters.direction === 'all' || c.direction === filters.direction) &&
      c.n_trades >= filters.minTrades &&
      (c.sqn || 0) >= filters.minSqn &&
      (c.sharpe || 0) >= filters.minSharpe
    )
    .sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (av === bv) return 0;
      return sortDir * (av > bv ? 1 : -1);
    }),
  [combos, filters, sortKey, sortDir]);

  const sqnColor = (v) => v >= 2 ? '#22c55e' : v >= 1 ? '#f59e0b' : '#ef4444';
  const kellyColor = (v) => {
    if (v < 0) return '#ef4444';
    if (v > 1) return '#f59e0b';
    return '#8890a4';
  };

  const arrow = (k) => sortKey === k ? (sortDir < 0 ? ' ▼' : ' ▲') : '';
  const TH = ({ label, k }) => (
    <th onClick={() => handleSort(k)} style={{ padding: '6px 8px', textAlign: 'right', color: sortKey === k ? '#3b82f6' : '#505870', fontSize: 10, fontFamily: FONT, fontWeight: 400, borderBottom: '1px solid #1e2230', cursor: 'pointer', whiteSpace: 'nowrap', userSelect: 'none', background: '#0f1014' }}>
      {label}{arrow(k)}
    </th>
  );
  const selectStyle = { background: '#111318', color: '#8890a4', border: '1px solid #1e2230', borderRadius: 4, padding: '4px 8px', fontSize: 11, fontFamily: FONT };

  const exportTable = () => {
    const cols = ['rank', 'symbol', 'horizon', 'threshold', 'direction', 'entry_prob', 'n_trades', 'p_win', 'avg_win', 'avg_loss', 'total_return', 'sharpe', 'sqn', 'ev', 'kelly'];
    downloadCsv(filtered, cols, `strategy_${selectedRun || 'run'}.csv`);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 11, color: '#505870', fontFamily: FONT }}>Run:</span>
        <select style={selectStyle} value={selectedRun} onChange={e => setSelectedRun(e.target.value)}>
          {runs.map(r => <option key={r} value={r}>{r}</option>)}
        </select>
        {meta && (
          <>
            <span style={{ fontSize: 10, color: '#3a3f52', fontFamily: FONT }}>
              {meta.test_range?.start} to {meta.test_range?.end}
            </span>
            <span style={{ fontSize: 10, color: '#3a3f52', fontFamily: FONT }}>
              Entry probs: [{(meta.entry_prob_grid || []).join(', ')}]
            </span>
          </>
        )}
      </div>

      {loading && <div style={{ color: '#505870', fontFamily: FONT, fontSize: 12, padding: 20 }}>Loading strategy data...</div>}
      {error && <div style={{ color: '#ef4444', fontFamily: FONT, fontSize: 12, padding: 20 }}>Error: {error} — Run a backtest first.</div>}

      {!loading && !error && (
        <>
          <div style={{ background: '#111318', border: '1px solid #1e2230', borderRadius: 8, padding: '12px 16px', display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center' }}>
            <span style={{ fontSize: 11, color: '#505870', fontFamily: FONT, marginRight: 4 }}>FILTER</span>
            <select style={selectStyle} value={filters.symbol} onChange={e => setFilters(f => ({ ...f, symbol: e.target.value }))}>
              <option value="all">All Symbols</option>
              {opts.symbols.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <select style={selectStyle} value={filters.horizon} onChange={e => setFilters(f => ({ ...f, horizon: e.target.value }))}>
              <option value="all">All Horizons</option>
              {opts.horizons.map(h => <option key={h} value={h}>{h}h</option>)}
            </select>
            <select style={selectStyle} value={filters.threshold} onChange={e => setFilters(f => ({ ...f, threshold: e.target.value }))}>
              <option value="all">All Thresholds</option>
              {opts.thresholds.map(t => <option key={t} value={t}>{t}%</option>)}
            </select>
            <select style={selectStyle} value={filters.direction} onChange={e => setFilters(f => ({ ...f, direction: e.target.value }))}>
              <option value="all">All Directions</option>
              {opts.directions.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
            <label style={{ fontSize: 11, color: '#505870', fontFamily: FONT, display: 'flex', alignItems: 'center', gap: 6 }}>
              Min Trades
              <input type="number" min="0" max="500" value={filters.minTrades}
                onChange={e => setFilters(f => ({ ...f, minTrades: +e.target.value }))}
                style={{ ...selectStyle, width: 56 }} />
            </label>
            <label style={{ fontSize: 11, color: '#505870', fontFamily: FONT, display: 'flex', alignItems: 'center', gap: 6 }}>
              Min SQN
              <input type="number" step="0.1" value={filters.minSqn}
                onChange={e => setFilters(f => ({ ...f, minSqn: +e.target.value }))}
                style={{ ...selectStyle, width: 56 }} />
            </label>
            <label style={{ fontSize: 11, color: '#505870', fontFamily: FONT, display: 'flex', alignItems: 'center', gap: 6 }}>
              Min Sharpe
              <input type="number" step="0.1" value={filters.minSharpe}
                onChange={e => setFilters(f => ({ ...f, minSharpe: +e.target.value }))}
                style={{ ...selectStyle, width: 56 }} />
            </label>
            <button onClick={exportTable} style={{ ...selectStyle, cursor: 'pointer' }}>↓ CSV</button>
            <span style={{ marginLeft: 'auto', fontSize: 11, color: '#505870', fontFamily: FONT }}>{filtered.length} combos</span>
          </div>

          <div style={{ background: '#111318', border: '1px solid #1e2230', borderRadius: 8, overflow: 'hidden' }}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 900 }}>
                <thead>
                  <tr>
                    <TH label="#" k="rank" />
                    <TH label="Symbol" k="symbol" />
                    <TH label="Hz" k="horizon" />
                    <TH label="Thr%" k="threshold" />
                    <TH label="Dir" k="direction" />
                    <TH label="Entry%" k="entry_prob" />
                    <TH label="N" k="n_trades" />
                    <TH label="P(Win)" k="p_win" />
                    <TH label="Avg+" k="avg_win" />
                    <TH label="Avg-" k="avg_loss" />
                    <TH label="Total%" k="total_return" />
                    <TH label="Sharpe" k="sharpe" />
                    <TH label="SQN" k="sqn" />
                    <TH label="EV%" k="ev" />
                    <TH label="Kelly" k="kelly" />
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((c, i) => {
                    const isSelected = selected?.rank === c.rank;
                    const td = (content, extra = {}) => (
                      <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: FONT, fontSize: 11, borderBottom: '1px solid #181b26', color: '#8890a4', ...extra }}>{content}</td>
                    );
                    return (
                      <tr key={c.rank} onClick={() => setSelected(c)}
                        style={{ background: isSelected ? '#14193a' : i % 2 === 0 ? '#0b0c0f' : '#0e0f13', cursor: 'pointer' }}
                        onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = '#131520'; }}
                        onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = i % 2 === 0 ? '#0b0c0f' : '#0e0f13'; }}>
                        {td(c.rank, { color: '#505870' })}
                        {td(c.symbol.replace('USDT', ''), { color: '#e2e4ea', fontWeight: 600 })}
                        {td(c.horizon + 'h')}
                        {td(c.threshold + '%')}
                        {td(c.direction, { color: c.direction === 'long' ? '#22c55e' : c.direction === 'short' ? '#ef4444' : '#8890a4' })}
                        {td(((c.entry_prob || 0) * 100).toFixed(0) + '%')}
                        {td((c.n_trades || 0) + (c.n_trades < 30 ? '*' : ''), { color: c.n_trades < 30 ? '#f59e0b' : '#8890a4' })}
                        {td(((c.p_win || 0) * 100).toFixed(1) + '%')}
                        {td('+' + Math.abs(c.avg_win || 0).toFixed(2) + '%', { color: '#22c55e' })}
                        {td('-' + Math.abs(c.avg_loss || 0).toFixed(2) + '%', { color: '#ef4444' })}
                        {td((c.total_return >= 0 ? '+' : '') + (c.total_return || 0) + '%', { color: c.total_return >= 0 ? '#22c55e' : '#ef4444', fontWeight: 600 })}
                        {td(c.sharpe || 0)}
                        {td(c.sqn || 0, { color: sqnColor(c.sqn || 0), fontWeight: 700, fontSize: 12 })}
                        {td((c.ev >= 0 ? '+' : '') + (c.ev || 0) + '%', { color: c.ev >= 0 ? '#22c55e' : '#ef4444' })}
                        {td(((c.kelly || 0) * 100).toFixed(1) + '%', { color: kellyColor(c.kelly || 0) })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div style={{ padding: '6px 12px', borderTop: '1px solid #1e2230', fontSize: 10, color: '#3a3f52', fontFamily: FONT }}>
              * n &lt; 30 — low sample size. Click any row to inspect combo. Kelly: red&lt;0, amber&gt;1 (over-bet).
            </div>
          </div>
        </>
      )}

      {selected && <DetailPanel combo={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
