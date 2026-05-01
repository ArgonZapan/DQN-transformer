import { useEffect, useMemo, useState } from 'react';
import OHLCVChart from './OHLCVChart.jsx';
import { ScatterChart, BarChart } from './charts.jsx';

const HORIZONS = [8, 12, 24, 48, 72];
const QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90];
const HORIZON_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#22c55e', '#ef4444'];
const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'AVAXUSDT', 'DOGEUSDT'];

const cardStyle = { background: '#111318', border: '1px solid #1e2230', borderRadius: 8, padding: 16 };
const cardTitle = { fontSize: 12, fontFamily: 'IBM Plex Mono, monospace', color: '#505870', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 };
const MONO = 'IBM Plex Mono, monospace';

function errColor(e) {
  const abs = Math.abs(e);
  if (abs < 3) return '#22c55e';
  if (abs < 8) return '#f59e0b';
  return '#ef4444';
}

function KpiTile({ label, value, sub, color = '#c5c9d6' }) {
  return (
    <div style={{ ...cardStyle, padding: 14 }}>
      <div style={{ ...cardTitle, marginBottom: 6 }}>{label}</div>
      <div style={{ fontFamily: MONO, fontSize: 22, color, fontWeight: 600 }}>{value}</div>
      {sub && <div style={{ fontFamily: MONO, fontSize: 10, color: '#3a3f52', marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function KpiRow({ calData }) {
  const stats = useMemo(() => {
    const rows = calData?.threshold_calibration ?? [];
    if (!rows.length) return null;
    const errs = rows.map(r => Math.abs(r.error_pct ?? 0));
    const mae = errs.reduce((a, b) => a + b, 0) / errs.length;
    const maxErr = Math.max(...errs);
    const greenCount = errs.filter(e => e < 3).length;
    const totalN = rows.reduce((a, r) => a + (r.n || 0), 0);
    const longBias = (() => {
      const longs = rows.filter(r => r.direction === 'long');
      if (!longs.length) return 0;
      return longs.reduce((a, r) => a + (r.error_pct || 0), 0) / longs.length;
    })();
    const shortBias = (() => {
      const shorts = rows.filter(r => r.direction === 'short');
      if (!shorts.length) return 0;
      return shorts.reduce((a, r) => a + (r.error_pct || 0), 0) / shorts.length;
    })();
    return { mae, maxErr, greenPct: (greenCount / errs.length) * 100, totalN, longBias, shortBias };
  }, [calData]);

  if (!stats) return null;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
      <KpiTile label="Mean Abs Error" value={`${stats.mae.toFixed(2)}%`} color={errColor(stats.mae)} sub="lower is better" />
      <KpiTile label="Max Error" value={`${stats.maxErr.toFixed(2)}%`} color={errColor(stats.maxErr)} sub="worst bucket" />
      <KpiTile label="Well Calibrated" value={`${stats.greenPct.toFixed(0)}%`} color={stats.greenPct > 60 ? '#22c55e' : stats.greenPct > 30 ? '#f59e0b' : '#ef4444'} sub="buckets with |err|<3%" />
      <KpiTile label="Long Bias" value={`${stats.longBias > 0 ? '+' : ''}${stats.longBias.toFixed(2)}%`} color={errColor(stats.longBias)} sub="signed avg error" />
      <KpiTile label="Short Bias" value={`${stats.shortBias > 0 ? '+' : ''}${stats.shortBias.toFixed(2)}%`} color={errColor(stats.shortBias)} sub="signed avg error" />
    </div>
  );
}

function CalibrationChart({ calData, direction, activeHorizons }) {
  const points = useMemo(() => calData.threshold_calibration.filter(d => d.direction === direction), [calData, direction]);

  const { xMax, yMax } = useMemo(() => {
    const xs = points.map(p => p.mean_pred_pct);
    const ys = points.map(p => p.actual_pct);
    const m = Math.max(10, ...xs, ...ys) * 1.1;
    return { xMax: Math.ceil(m / 10) * 10, yMax: Math.ceil(m / 10) * 10 };
  }, [points]);

  const series = HORIZONS.map((h, hi) => ({
    label: `${h}h`,
    color: HORIZON_COLORS[hi],
    points: activeHorizons.has(h)
      ? points.filter(d => d.horizon_h === h).map(d => ({
          x: d.mean_pred_pct, y: d.actual_pct,
          tip: `thr:${d.threshold_pct}% pred:${d.mean_pred_pct}% actual:${d.actual_pct}% n:${d.n}`,
        }))
      : [],
  }));

  return (
    <ScatterChart
      width={500} height={240} series={series}
      xLabel="Predicted %" yLabel="Actual %"
      xMin={0} xMax={xMax} yMin={0} yMax={yMax} diagonal
    />
  );
}

function QuantileCoveragePerHorizon({ calData, direction }) {
  const filtered = calData.quantile_coverage.filter(d => d.direction === direction);
  const labels = QUANTILES.map(q => `p${Math.round(q * 100)}`);
  const expected = QUANTILES.map(q => q * 100);
  const actualColor = direction === 'long' ? 'rgba(34,197,94,0.5)' : 'rgba(239,68,68,0.5)';
  const actualBorder = direction === 'long' ? '#22c55e' : '#ef4444';

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8 }}>
      {HORIZONS.map((h, hi) => {
        const rows = filtered.filter(d => d.horizon_h === h);
        const actual = QUANTILES.map(q => {
          const r = rows.find(d => d.quantile === q);
          return r ? r.actual_pct : null;
        });
        return (
          <div key={h} style={{ background: '#0c0d11', border: `1px solid ${HORIZON_COLORS[hi]}33`, borderRadius: 6, padding: 8 }}>
            <div style={{ fontSize: 10, fontFamily: MONO, color: HORIZON_COLORS[hi], marginBottom: 4, fontWeight: 600 }}>{h}h</div>
            <BarChart
              width={220} height={110} labels={labels} yMin={0} yMax={100}
              groups={[
                { label: 'Expected', color: 'rgba(59,130,246,0.4)', border: '#3b82f6', values: expected },
                { label: 'Actual', color: actualColor, border: actualBorder, values: actual },
              ]}
            />
          </div>
        );
      })}
    </div>
  );
}

function downloadCsv(rows, filename) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const lines = [cols.join(',')];
  rows.forEach(r => lines.push(cols.map(c => JSON.stringify(r[c] ?? '')).join(',')));
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function SummaryTable({ calData, runId }) {
  const [dirFilter, setDirFilter] = useState('long');
  const [hFilter, setHFilter] = useState('all');
  const [sortKey, setSortKey] = useState(null);
  const [sortDir, setSortDir] = useState('desc');

  const rows = useMemo(() => {
    if (!calData) return [];
    const base = calData.threshold_calibration.filter(d =>
      d.direction === dirFilter && (hFilter === 'all' || d.horizon_h === +hFilter)
    );
    if (!sortKey) return base;
    return [...base].sort((a, b) => {
      const av = sortKey === 'abs_error' ? Math.abs(a.error_pct ?? 0) : a[sortKey] ?? 0;
      const bv = sortKey === 'abs_error' ? Math.abs(b.error_pct ?? 0) : b[sortKey] ?? 0;
      return sortDir === 'asc' ? av - bv : bv - av;
    });
  }, [calData, dirFilter, hFilter, sortKey, sortDir]);

  if (!calData) return null;

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  const S = {
    th: { padding: '6px 10px', textAlign: 'right', color: '#505870', fontSize: 11, fontFamily: MONO, fontWeight: 400, whiteSpace: 'nowrap', borderBottom: '1px solid #1e2230', cursor: 'pointer', userSelect: 'none' },
    td: { padding: '5px 10px', textAlign: 'right', fontSize: 11, fontFamily: MONO, borderBottom: '1px solid #181b26', color: '#c5c9d6' },
  };

  const arrow = (key) => sortKey === key ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '';

  const cols = [
    { key: 'horizon_h', label: 'Horizon' },
    { key: 'threshold_pct', label: 'Threshold%' },
    { key: 'mean_pred_pct', label: 'Mean Pred%' },
    { key: 'actual_pct', label: 'Actual%' },
    { key: 'abs_error', label: 'Error%' },
    { key: 'n', label: 'N' },
  ];

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10, alignItems: 'center' }}>
        {['long', 'short'].map(d => (
          <button key={d} onClick={() => setDirFilter(d)} style={{
            padding: '3px 12px', fontSize: 11, fontFamily: MONO,
            background: dirFilter === d ? (d === 'long' ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)') : 'transparent',
            color: dirFilter === d ? (d === 'long' ? '#22c55e' : '#ef4444') : '#505870',
            border: '1px solid ' + (dirFilter === d ? (d === 'long' ? '#22c55e' : '#ef4444') : '#1e2230'),
            borderRadius: 4, cursor: 'pointer',
          }}>{d.toUpperCase()}</button>
        ))}
        <select value={hFilter} onChange={e => setHFilter(e.target.value)} style={{
          marginLeft: 8, background: '#111318', color: '#8890a4', border: '1px solid #1e2230',
          borderRadius: 4, padding: '3px 8px', fontSize: 11, fontFamily: MONO,
        }}>
          <option value="all">All Horizons</option>
          {HORIZONS.map(h => <option key={h} value={h}>{h}h</option>)}
        </select>
        <button
          onClick={() => downloadCsv(rows, `calibration_${runId || 'run'}_${dirFilter}_${hFilter}.csv`)}
          style={{
            marginLeft: 'auto', padding: '3px 10px', fontSize: 11, fontFamily: MONO,
            background: 'transparent', color: '#8890a4', border: '1px solid #1e2230',
            borderRadius: 4, cursor: 'pointer',
          }}
        >
          ↓ CSV
        </button>
      </div>
      <div style={{ overflowX: 'auto', borderRadius: 6, border: '1px solid #1e2230' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#0f1014' }}>
              {cols.map(c => (
                <th key={c.key} style={S.th} onClick={() => toggleSort(c.key)}>
                  {c.label}{arrow(c.key)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? '#0b0c0f' : '#0e0f13' }}>
                <td style={{ ...S.td, color: HORIZON_COLORS[HORIZONS.indexOf(r.horizon_h)] }}>{r.horizon_h}h</td>
                <td style={S.td}>{r.threshold_pct}%</td>
                <td style={S.td}>{r.mean_pred_pct}%</td>
                <td style={S.td}>{r.actual_pct}%</td>
                <td style={{ ...S.td, color: errColor(r.error_pct), fontWeight: 600 }}>
                  {r.error_pct > 0 ? '+' : ''}{r.error_pct}%
                </td>
                <td style={S.td}>{(r.n || 0).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function BacktestSection() {
  const [calData, setCalData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState('');
  const [calDir, setCalDir] = useState('long');
  const [activeHorizons, setActiveHorizons] = useState(new Set(HORIZONS));
  const [symbol, setSymbol] = useState('BTCUSDT');

  useEffect(() => {
    fetch('/api/runs')
      .then(r => r.json())
      .then(data => {
        setRuns(data);
        if (data.length > 0) setSelectedRun(data[data.length - 1]);
        else setLoading(false);
      })
      .catch(e => { setError(String(e)); setLoading(false); });
  }, []);

  useEffect(() => {
    if (!selectedRun) return;
    setLoading(true);
    setError(null);
    fetch(`/api/calibration/${selectedRun}`)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(data => { setCalData(data); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [selectedRun]);

  const toggleHorizon = (h) => {
    setActiveHorizons(prev => {
      const next = new Set(prev);
      if (next.has(h)) next.delete(h); else next.add(h);
      return next.size === 0 ? new Set(HORIZONS) : next;
    });
  };

  const Toolbar = (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
      <span style={{ fontSize: 11, color: '#505870', fontFamily: MONO }}>Run:</span>
      <select value={selectedRun} onChange={e => setSelectedRun(e.target.value)} disabled={!runs.length} style={{
        background: '#111318', color: '#8890a4', border: '1px solid #1e2230', borderRadius: 4,
        padding: '4px 8px', fontSize: 11, fontFamily: MONO, minWidth: 180,
      }}>
        {runs.length === 0 && <option>—</option>}
        {runs.map(r => <option key={r} value={r}>{r}</option>)}
      </select>
      <span style={{ fontSize: 11, color: '#505870', fontFamily: MONO, marginLeft: 12 }}>Symbol:</span>
      <select value={symbol} onChange={e => setSymbol(e.target.value)} style={{
        background: '#111318', color: '#8890a4', border: '1px solid #1e2230', borderRadius: 4,
        padding: '4px 8px', fontSize: 11, fontFamily: MONO,
      }}>
        {SYMBOLS.map(s => <option key={s} value={s}>{s}</option>)}
      </select>
      {calData?.generated_at && (
        <span style={{ fontSize: 10, color: '#3a3f52', fontFamily: MONO, marginLeft: 'auto' }}>
          Generated: {new Date(calData.generated_at).toLocaleString()}
        </span>
      )}
    </div>
  );

  const errorBlock = error ? (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: 200, gap: 10 }}>
      <div style={{ color: '#ef4444', fontFamily: MONO, fontSize: 13 }}>Failed to load: {error}</div>
      <div style={{ color: '#505870', fontFamily: MONO, fontSize: 11 }}>
        Run <code>python -m python.server.sse_server</code> and run a backtest first.
      </div>
    </div>
  ) : null;

  const emptyBlock = !calData ? (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200, color: '#505870', fontFamily: MONO, fontSize: 12 }}>
      {loading ? 'Loading calibration data…' : 'No data.'}
    </div>
  ) : null;

  const body = calData ? (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20, position: 'relative' }}>
        {loading && (
          <div style={{ position: 'absolute', top: -4, right: 0, fontSize: 10, color: '#3b82f6', fontFamily: MONO }}>
            ⟳ refreshing…
          </div>
        )}

        <KpiRow calData={calData} />

        <div style={cardStyle}>
          <div style={cardTitle}>Market Context — OHLCV ({symbol})</div>
          <OHLCVChart symbol={symbol} interval="1d" height={520} showVolume />
        </div>

        <div style={cardStyle}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
            <span style={cardTitle}>Threshold Calibration</span>
            <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
              {HORIZONS.map((h, hi) => {
                const active = activeHorizons.has(h);
                return (
                  <button key={h} onClick={() => toggleHorizon(h)} style={{
                    padding: '2px 8px', fontSize: 10, fontFamily: MONO,
                    background: active ? `${HORIZON_COLORS[hi]}22` : 'transparent',
                    color: active ? HORIZON_COLORS[hi] : '#3a3f52',
                    border: `1px solid ${active ? HORIZON_COLORS[hi] : '#1e2230'}`,
                    borderRadius: 3, cursor: 'pointer',
                  }}>{h}h</button>
                );
              })}
              <span style={{ width: 8 }} />
              {['long', 'short'].map(d => (
                <button key={d} onClick={() => setCalDir(d)} style={{
                  padding: '2px 9px', fontSize: 10, fontFamily: MONO,
                  background: calDir === d ? (d === 'long' ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)') : 'transparent',
                  color: calDir === d ? (d === 'long' ? '#22c55e' : '#ef4444') : '#505870',
                  border: '1px solid ' + (calDir === d ? (d === 'long' ? '#22c55e' : '#ef4444') : '#1e2230'),
                  borderRadius: 3, cursor: 'pointer',
                }}>{d}</button>
              ))}
            </div>
          </div>
          <CalibrationChart calData={calData} direction={calDir} activeHorizons={activeHorizons} />
          <p style={{ fontSize: 10, color: '#3a3f52', marginTop: 8, fontFamily: MONO }}>
            Dots above diagonal = model overestimates. Below = underestimates. Click horizon chips to toggle.
          </p>
        </div>

        {['long', 'short'].map(dir => (
          <div key={dir} style={cardStyle}>
            <div style={cardTitle}>Quantile Coverage — {dir} (per horizon)</div>
            <QuantileCoveragePerHorizon calData={calData} direction={dir} />
            <p style={{ fontSize: 10, color: '#3a3f52', marginTop: 8, fontFamily: MONO }}>
              Blue = expected coverage (e.g. p90 → 90%). Coloured = actual. Missing bars = no data for that bucket.
            </p>
          </div>
        ))}

        <div style={cardStyle}>
          <div style={cardTitle}>Calibration Summary Table</div>
          <SummaryTable calData={calData} runId={selectedRun} />
        </div>
      </div>
  ) : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {Toolbar}
      {errorBlock || emptyBlock || body}
    </div>
  );
}
