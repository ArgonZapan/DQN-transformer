// Backtest Explorer — Section 1
// Calibration charts + quantile coverage + summary table
// Data: fetched from /api/calibration (served by python/server/sse_server.py)

const HORIZONS = [8, 12, 24, 48, 72];
const THRESHOLDS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0];
const QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90];
const HORIZON_COLORS = ['#3b82f6','#8b5cf6','#f59e0b','#22c55e','#ef4444'];

// ── Chart components ────────────────────────────────────────────────────────

function CalibrationChart({ calData, direction }) {
  const ref = React.useRef(null);
  const inst = React.useRef(null);

  React.useEffect(() => {
    if (!ref.current || !calData) return;
    const ctx = ref.current.getContext('2d');
    if (inst.current) inst.current.destroy();

    const filtered = calData.threshold_calibration.filter(d => d.direction === direction);
    const datasets = HORIZONS.map((h, hi) => {
      const pts = filtered
        .filter(d => d.horizon_h === h)
        .sort((a,b) => a.mean_pred_pct - b.mean_pred_pct)
        .map(d => ({ x: d.mean_pred_pct, y: d.actual_pct, thr: d.threshold_pct }));
      return {
        label: `${h}h`, data: pts,
        borderColor: HORIZON_COLORS[hi], backgroundColor: HORIZON_COLORS[hi],
        pointRadius: 5, pointHoverRadius: 7, borderWidth: 1.5, tension: 0.2, showLine: true,
      };
    });
    datasets.push({
      label: 'Perfect', data: [{ x: 0, y: 0 }, { x: 80, y: 80 }],
      borderColor: 'rgba(255,255,255,0.2)', backgroundColor: 'transparent',
      pointRadius: 0, borderWidth: 1, borderDash: [4, 4], showLine: true,
    });

    inst.current = new Chart(ctx, {
      type: 'scatter', data: { datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#8890a4', font: { size: 10, family: 'IBM Plex Mono' }, boxWidth: 12 } },
          tooltip: { callbacks: { label: (item) => {
            const d = item.raw;
            return `${item.dataset.label} | thr:${d.thr}% pred:${d.x}% actual:${d.y}%`;
          }}}
        },
        scales: {
          x: { title: { display: true, text: 'Predicted %', color: '#505870', font: { size: 10 } },
            ticks: { color: '#505870', font: { size: 9, family: 'IBM Plex Mono' } },
            grid: { color: '#181b26' }, min: 0, max: 80 },
          y: { title: { display: true, text: 'Actual %', color: '#505870', font: { size: 10 } },
            ticks: { color: '#505870', font: { size: 9, family: 'IBM Plex Mono' } },
            grid: { color: '#181b26' }, min: 0, max: 80 },
        },
      },
    });
    return () => inst.current?.destroy();
  }, [calData, direction]);

  return <canvas ref={ref} />;
}

function QuantileCoverageChart({ calData, direction }) {
  const ref = React.useRef(null);
  const inst = React.useRef(null);

  React.useEffect(() => {
    if (!ref.current || !calData) return;
    const ctx = ref.current.getContext('2d');
    if (inst.current) inst.current.destroy();

    const filtered = calData.quantile_coverage.filter(d => d.direction === direction);
    const byQ = {};
    QUANTILES.forEach(q => { byQ[q] = []; });
    filtered.forEach(d => { if (byQ[d.quantile] !== undefined) byQ[d.quantile].push(d.actual_pct); });

    const labels = QUANTILES.map(q => `p${Math.round(q*100)}`);
    const expected = QUANTILES.map(q => q * 100);
    const actual = QUANTILES.map(q => {
      const vals = byQ[q] || [];
      return vals.length ? +(vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(1) : 0;
    });

    inst.current = new Chart(ctx, {
      type: 'bar',
      data: { labels, datasets: [
        { label: 'Expected', data: expected, backgroundColor: 'rgba(59,130,246,0.3)', borderColor: '#3b82f6', borderWidth: 1 },
        { label: 'Actual', data: actual,
          backgroundColor: direction === 'long' ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)',
          borderColor: direction === 'long' ? '#22c55e' : '#ef4444', borderWidth: 1 },
      ]},
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#8890a4', font: { size: 10, family: 'IBM Plex Mono' }, boxWidth: 12 } } },
        scales: {
          x: { ticks: { color: '#505870', font: { size: 10, family: 'IBM Plex Mono' } }, grid: { color: '#181b26' } },
          y: { ticks: { color: '#505870', font: { size: 9, family: 'IBM Plex Mono' } }, grid: { color: '#181b26' }, min: 0, max: 100 },
        },
      },
    });
    return () => inst.current?.destroy();
  }, [calData, direction]);

  return <canvas ref={ref} />;
}

function SummaryTable({ calData }) {
  const [dirFilter, setDirFilter] = React.useState('long');
  const [hFilter, setHFilter] = React.useState('all');

  if (!calData) return null;

  const rows = calData.threshold_calibration.filter(d =>
    d.direction === dirFilter && (hFilter === 'all' || d.horizon_h === +hFilter)
  );

  const errColor = (e) => {
    const abs = Math.abs(e);
    if (abs < 3) return '#22c55e';
    if (abs < 8) return '#f59e0b';
    return '#ef4444';
  };

  const S = {
    th: { padding: '6px 10px', textAlign: 'right', color: '#505870', fontSize: 11, fontFamily: 'IBM Plex Mono, monospace', fontWeight: 400, whiteSpace: 'nowrap', borderBottom: '1px solid #1e2230' },
    td: { padding: '5px 10px', textAlign: 'right', fontSize: 11, fontFamily: 'IBM Plex Mono, monospace', borderBottom: '1px solid #181b26', color: '#c5c9d6' },
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        {['long','short'].map(d => (
          <button key={d} onClick={() => setDirFilter(d)} style={{
            padding: '3px 12px', fontSize: 11, fontFamily: 'IBM Plex Mono, monospace',
            background: dirFilter === d ? (d === 'long' ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)') : 'transparent',
            color: dirFilter === d ? (d === 'long' ? '#22c55e' : '#ef4444') : '#505870',
            border: '1px solid ' + (dirFilter === d ? (d === 'long' ? '#22c55e' : '#ef4444') : '#1e2230'),
            borderRadius: 4, cursor: 'pointer',
          }}>{d.toUpperCase()}</button>
        ))}
        <select value={hFilter} onChange={e => setHFilter(e.target.value)} style={{
          marginLeft: 8, background: '#111318', color: '#8890a4', border: '1px solid #1e2230',
          borderRadius: 4, padding: '3px 8px', fontSize: 11, fontFamily: 'IBM Plex Mono, monospace',
        }}>
          <option value="all">All Horizons</option>
          {HORIZONS.map(h => <option key={h} value={h}>{h}h</option>)}
        </select>
      </div>
      <div style={{ overflowX: 'auto', borderRadius: 6, border: '1px solid #1e2230' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#0f1014' }}>
              {['Horizon','Threshold%','Mean Pred%','Actual%','Error%','N'].map(h => (
                <th key={h} style={S.th}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} style={{ background: i%2===0 ? '#0b0c0f' : '#0e0f13' }}>
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

// ── Main Section ─────────────────────────────────────────────────────────────
const BacktestSection = () => {
  const [calData, setCalData] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError]     = React.useState(null);
  const [runs, setRuns]       = React.useState([]);
  const [selectedRun, setSelectedRun] = React.useState('');
  const [calDir, setCalDir]   = React.useState('long');

  // Load available runs
  React.useEffect(() => {
    fetch('/api/runs')
      .then(r => r.json())
      .then(data => {
        setRuns(data);
        if (data.length > 0) setSelectedRun(data[data.length - 1]);
      })
      .catch(() => {});
  }, []);

  // Load calibration data when run changes
  React.useEffect(() => {
    if (!selectedRun) return;
    setLoading(true);
    setError(null);
    fetch(`/api/calibration/${selectedRun}`)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(data => { setCalData(data); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [selectedRun]);

  const cardStyle = { background: '#111318', border: '1px solid #1e2230', borderRadius: 8, padding: 16 };
  const cardTitle = { fontSize: 12, fontFamily: 'IBM Plex Mono, monospace', color: '#505870', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 };

  if (loading) return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:200, color:'#505870', fontFamily:'IBM Plex Mono,monospace', fontSize:12 }}>
      Loading calibration data...
    </div>
  );

  if (error) return (
    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height:200, gap:10 }}>
      <div style={{ color:'#ef4444', fontFamily:'IBM Plex Mono,monospace', fontSize:13 }}>
        Failed to load: {error}
      </div>
      <div style={{ color:'#505870', fontFamily:'IBM Plex Mono,monospace', fontSize:11 }}>
        Run <code>python -m python.server.sse_server</code> and run a backtest first.
      </div>
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Run selector */}
      <div style={{ display:'flex', alignItems:'center', gap:10 }}>
        <span style={{ fontSize:11, color:'#505870', fontFamily:'IBM Plex Mono,monospace' }}>Run:</span>
        <select value={selectedRun} onChange={e => setSelectedRun(e.target.value)} style={{
          background:'#111318', color:'#8890a4', border:'1px solid #1e2230', borderRadius:4,
          padding:'4px 8px', fontSize:11, fontFamily:'IBM Plex Mono,monospace',
        }}>
          {runs.map(r => <option key={r} value={r}>{r}</option>)}
        </select>
        {calData?.generated_at && (
          <span style={{ fontSize:10, color:'#3a3f52', fontFamily:'IBM Plex Mono,monospace' }}>
            Generated: {new Date(calData.generated_at).toLocaleString()}
          </span>
        )}
      </div>

      {/* OHLCV Chart */}
      <div style={cardStyle}>
        <div style={cardTitle}>Market Context — OHLCV</div>
        <OHLCVChart symbol="BTCUSDT" interval="1d" height={280} showVolume={true} />
      </div>

      {/* Calibration Charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div style={cardStyle}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <span style={cardTitle}>Threshold Calibration</span>
            <div style={{ display: 'flex', gap: 4 }}>
              {['long','short'].map(d => (
                <button key={d} onClick={() => setCalDir(d)} style={{
                  padding: '2px 9px', fontSize: 10, fontFamily: 'IBM Plex Mono, monospace',
                  background: calDir===d ? (d==='long'?'rgba(34,197,94,0.15)':'rgba(239,68,68,0.15)') : 'transparent',
                  color: calDir===d ? (d==='long'?'#22c55e':'#ef4444') : '#505870',
                  border: '1px solid '+(calDir===d?(d==='long'?'#22c55e':'#ef4444'):'#1e2230'),
                  borderRadius: 3, cursor: 'pointer',
                }}>{d}</button>
              ))}
            </div>
          </div>
          <div style={{ height: 220 }}>
            <CalibrationChart key={calDir} calData={calData} direction={calDir} />
          </div>
          <p style={{ fontSize: 10, color: '#3a3f52', marginTop: 8, fontFamily: 'IBM Plex Mono, monospace' }}>
            Dots above diagonal = model overestimates. Below = underestimates.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateRows: '1fr 1fr', gap: 12 }}>
          {['long','short'].map(dir => (
            <div key={dir} style={cardStyle}>
              <div style={cardTitle}>Quantile Coverage — {dir}</div>
              <div style={{ height: 130 }}>
                <QuantileCoverageChart calData={calData} direction={dir} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Summary Table */}
      <div style={cardStyle}>
        <div style={cardTitle}>Calibration Summary Table</div>
        <SummaryTable calData={calData} />
      </div>
    </div>
  );
};

Object.assign(window, { BacktestSection });
