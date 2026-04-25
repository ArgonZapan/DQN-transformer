// Live Predictions — Section 3
// Symbol list + OHLCV fan overlay + probability chart + signal status
// Data: SSE stream from /api/live (served by python/server/sse_server.py)

const LIVE_SYMBOLS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT'];
const HORIZONS_H = [8, 12, 24, 48, 72];
const THRESHOLDS_PCT = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0];

// ── Probability bar chart ────────────────────────────────────────────────────
function ProbabilityChart({ horizonData, entryProbLine }) {
  const ref = React.useRef(null);
  const inst = React.useRef(null);
  React.useEffect(() => {
    if (!ref.current || !horizonData) return;
    const ctx = ref.current.getContext('2d');
    if (inst.current) inst.current.destroy();
    const keys = Object.keys(horizonData.long.threshold_probs);
    const longVals  = keys.map(k => ((horizonData.long.threshold_probs[k]||0)*100).toFixed(1));
    const shortVals = keys.map(k => ((horizonData.short.threshold_probs[k]||0)*100).toFixed(1));
    inst.current = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: keys.map(k => k.replace('pct','%')),
        datasets: [
          { label: 'Long', data: longVals, backgroundColor: 'rgba(34,197,94,0.45)', borderColor: '#22c55e', borderWidth: 1 },
          { label: 'Short', data: shortVals, backgroundColor: 'rgba(239,68,68,0.45)', borderColor: '#ef4444', borderWidth: 1 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#8890a4', font: { size: 10, family: 'IBM Plex Mono' }, boxWidth: 10 } },
        },
        scales: {
          x: { ticks:{color:'#505870',font:{size:10,family:'IBM Plex Mono'}}, grid:{color:'#181b26'} },
          y: { min:0, max:100, ticks:{color:'#505870',font:{size:9,family:'IBM Plex Mono'},callback:v=>v+'%'}, grid:{color:'#181b26'} },
        },
      },
    });
    return () => inst.current?.destroy();
  }, [horizonData, entryProbLine]);
  return <canvas ref={ref} />;
}

// ── Fan summary mini-viz (SVG) ────────────────────────────────────────────────
function QuantileFanSVG({ prediction, width=260, height=90 }) {
  if (!prediction) return null;
  const horizons = prediction.horizons;
  const hCount = horizons.length;
  const pad = { l:30, r:10, t:8, b:18 };
  const W = width - pad.l - pad.r;
  const H = height - pad.t - pad.b;

  const allVals = horizons.flatMap(h => [
    h.long?.quantiles?.p90 || 0, -(h.short?.quantiles?.p90 || 0), 0
  ]);
  const yMin = Math.min(...allVals) * 1.1;
  const yMax = Math.max(...allVals) * 1.1;
  const yRange = yMax - yMin || 1;

  const xOf = (i) => pad.l + (i / Math.max(1, hCount-1)) * W;
  const yOf = (v) => pad.t + H - ((v - yMin) / yRange) * H;
  const makePath = (vals) => vals.map((v,i) => `${i===0?'M':'L'}${xOf(i)},${yOf(v)}`).join(' ');

  const longQ  = (q) => horizons.map(h => h.long?.quantiles?.[q]  || 0);
  const shortQ = (q) => horizons.map(h => h.short?.quantiles?.[q] || 0);
  const zero = yOf(0);

  return (
    <svg width={width} height={height} style={{ display:'block' }}>
      <line x1={pad.l} x2={pad.l+W} y1={zero} y2={zero} stroke="#1e2230" strokeWidth={1} />
      <path d={`${makePath(longQ('p90'))} L${xOf(hCount-1)},${zero} L${xOf(0)},${zero} Z`} fill="rgba(34,197,94,0.07)" />
      <path d={`${makePath(longQ('p75'))} L${xOf(hCount-1)},${zero} L${xOf(0)},${zero} Z`} fill="rgba(34,197,94,0.12)" />
      <path d={`${makePath(longQ('p25'))} L${xOf(hCount-1)},${zero} L${xOf(0)},${zero} Z`} fill="rgba(34,197,94,0.18)" />
      <path d={makePath(longQ('p50'))} fill="none" stroke="rgba(34,197,94,0.85)" strokeWidth={1.5} />
      <path d={`${makePath(shortQ('p90').map(v=>-v))} L${xOf(hCount-1)},${zero} L${xOf(0)},${zero} Z`} fill="rgba(239,68,68,0.07)" />
      <path d={`${makePath(shortQ('p75').map(v=>-v))} L${xOf(hCount-1)},${zero} L${xOf(0)},${zero} Z`} fill="rgba(239,68,68,0.12)" />
      <path d={`${makePath(shortQ('p25').map(v=>-v))} L${xOf(hCount-1)},${zero} L${xOf(0)},${zero} Z`} fill="rgba(239,68,68,0.18)" />
      <path d={makePath(shortQ('p50').map(v=>-v))} fill="none" stroke="rgba(239,68,68,0.85)" strokeWidth={1.5} />
      {horizons.map((h,i) => (
        <text key={h.horizon_h} x={xOf(i)} y={height-2} textAnchor="middle" fontSize={8} fill="#3a3f52" fontFamily="IBM Plex Mono">{h.horizon_h}h</text>
      ))}
      <text x={pad.l-2} y={pad.t+4} textAnchor="end" fontSize={7} fill="#3a3f52" fontFamily="IBM Plex Mono">+{yMax.toFixed(1)}%</text>
      <text x={pad.l-2} y={height-pad.b+2} textAnchor="end" fontSize={7} fill="#3a3f52" fontFamily="IBM Plex Mono">{yMin.toFixed(1)}%</text>
    </svg>
  );
}

// ── Signal detector ──────────────────────────────────────────────────────────
function detectSignal(prediction, bestCombo) {
  if (!prediction || !bestCombo) return null;
  const { horizon, threshold, entry_prob, direction } = bestCombo;
  const horizonData = prediction.horizons.find(h => h.horizon_h === horizon);
  if (!horizonData) return null;
  const key = `${threshold}pct`;
  const dirs = direction === 'both' ? ['long','short'] : [direction];
  for (const dir of dirs) {
    const prob = horizonData[dir]?.threshold_probs?.[key];
    if (prob != null && prob >= entry_prob) return { dir, prob, horizon, threshold };
  }
  return null;
}

// ── Symbol card ──────────────────────────────────────────────────────────────
function SymbolCard({ pred, isSelected, onClick, signal }) {
  const price = pred.current_price;
  const fmt = (p) => p > 100 ? p.toLocaleString('en-US',{maximumFractionDigits:2}) : p.toFixed(4);
  const h24 = pred.horizons.find(h=>h.horizon_h===24);
  const p50L = h24?.long?.quantiles?.p50;
  const p50S = h24?.short?.quantiles?.p50;

  return (
    <div onClick={onClick} style={{
      padding:'10px 12px', cursor:'pointer', borderBottom:'1px solid #181b26',
      background: isSelected ? '#131a2e' : 'transparent',
      borderLeft: isSelected ? '2px solid #3b82f6' : '2px solid transparent',
    }}
    onMouseEnter={e=>{ if(!isSelected) e.currentTarget.style.background='#0f1118'; }}
    onMouseLeave={e=>{ if(!isSelected) e.currentTarget.style.background='transparent'; }}>
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:4 }}>
        <span style={{ fontFamily:'IBM Plex Mono,monospace', fontWeight:700, color:'#e2e4ea', fontSize:12 }}>
          {pred.symbol.replace('USDT','')}<span style={{ color:'#3a3f52', fontWeight:400 }}>/USDT</span>
        </span>
        {signal && (
          <span style={{ fontSize:9, fontFamily:'IBM Plex Mono,monospace', fontWeight:700,
            color: signal.dir==='long'?'#22c55e':'#ef4444',
            background: signal.dir==='long'?'rgba(34,197,94,0.12)':'rgba(239,68,68,0.12)',
            border:`1px solid ${signal.dir==='long'?'rgba(34,197,94,0.3)':'rgba(239,68,68,0.3)'}`,
            borderRadius:3, padding:'1px 5px' }}>
            {signal.dir.toUpperCase()}
          </span>
        )}
      </div>
      <div style={{ fontFamily:'IBM Plex Mono,monospace', color:'#c5c9d6', fontSize:13, marginBottom:6 }}>
        {fmt(price)}
      </div>
      <QuantileFanSVG prediction={pred} width={188} height={60} />
      {p50L != null && (
        <div style={{ display:'flex', gap:10, marginTop:4 }}>
          <span style={{ fontSize:9, color:'#22c55e', fontFamily:'IBM Plex Mono,monospace' }}>&#9650; {p50L}% 24h</span>
          <span style={{ fontSize:9, color:'#ef4444', fontFamily:'IBM Plex Mono,monospace' }}>&#9660; {p50S}% 24h</span>
        </div>
      )}
    </div>
  );
}

// ── Main Section ─────────────────────────────────────────────────────────────
const LiveSection = ({ bestCombo }) => {
  const [liveData, setLiveData]       = React.useState(null);
  const [selected, setSelected]       = React.useState('BTCUSDT');
  const [selectedHorizon, setSelectedHorizon] = React.useState(24);
  const [lastRefresh, setLastRefresh] = React.useState(null);
  const [sseStatus, setSseStatus]     = React.useState('connecting');

  // SSE connection
  React.useEffect(() => {
    const es = new EventSource('/api/live');
    es.onopen = () => setSseStatus('connected');
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setLiveData(data);
        setLastRefresh(new Date(data.generated_at));
        setSseStatus('connected');
      } catch (_) {}
    };
    es.onerror = () => setSseStatus('error');
    return () => es.close();
  }, []);

  // Manual refresh: one-shot fetch
  const refresh = async () => {
    setSseStatus('refreshing');
    try {
      const data = await fetch('/api/live').then(r => r.json());
      setLiveData(data);
      setLastRefresh(new Date(data.generated_at));
      setSseStatus('connected');
    } catch (e) {
      setSseStatus('error');
    }
  };

  const selPred = liveData?.predictions?.find(p => p.symbol === selected);
  const selHorizonData = selPred?.horizons?.find(h => h.horizon_h === selectedHorizon);

  const fanData = selPred ? {
    currentPrice: selPred.current_price,
    horizons: selPred.horizons.map(h => ({
      h: h.horizon_h,
      long:  h.long?.quantiles  || {},
      short: h.short?.quantiles || {},
    })),
  } : null;

  const signals = (liveData?.predictions || []).map(p => ({
    symbol: p.symbol, signal: detectSignal(p, bestCombo),
  }));

  const statusColor = { connected:'#22c55e', connecting:'#f59e0b', error:'#ef4444', refreshing:'#3b82f6' };
  const cardS = { background:'#111318', border:'1px solid #1e2230', borderRadius:8, padding:14 };
  const titleS = { fontSize:10, color:'#505870', fontFamily:'IBM Plex Mono,monospace', textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:10 };

  return (
    <div style={{ display:'flex', gap:16, height:'calc(100vh - 120px)', minHeight:600 }}>
      {/* Left: Symbol list */}
      <div style={{ width:212, flexShrink:0, background:'#111318', border:'1px solid #1e2230', borderRadius:8, overflow:'hidden', display:'flex', flexDirection:'column' }}>
        <div style={{ padding:'10px 12px', borderBottom:'1px solid #1e2230', display:'flex', alignItems:'center', justifyContent:'space-between' }}>
          <span style={{ fontSize:10, color:'#505870', fontFamily:'IBM Plex Mono,monospace', textTransform:'uppercase', letterSpacing:'0.08em' }}>Symbols</span>
          <div style={{ display:'flex', alignItems:'center', gap:6 }}>
            <span style={{ width:6, height:6, borderRadius:'50%', background:statusColor[sseStatus]||'#6b7280', display:'inline-block' }} />
            <button onClick={refresh} style={{ background:'transparent', border:'1px solid #1e2230', borderRadius:3, color:'#8890a4', cursor:'pointer', fontSize:9, fontFamily:'IBM Plex Mono,monospace', padding:'2px 6px' }}>
              &#x21BB;
            </button>
          </div>
        </div>
        <div style={{ flex:1, overflowY:'auto' }}>
          {liveData ? liveData.predictions.map(p => (
            <SymbolCard key={p.symbol} pred={p} isSelected={p.symbol === selected}
              onClick={() => setSelected(p.symbol)}
              signal={signals.find(s=>s.symbol===p.symbol)?.signal} />
          )) : LIVE_SYMBOLS.map(s => (
            <div key={s} style={{ padding:'10px 12px', borderBottom:'1px solid #181b26', color:'#3a3f52', fontSize:12, fontFamily:'IBM Plex Mono,monospace' }}>{s}</div>
          ))}
        </div>
        <div style={{ padding:'8px 12px', borderTop:'1px solid #1e2230', fontSize:9, color:'#3a3f52', fontFamily:'IBM Plex Mono,monospace' }}>
          {sseStatus === 'connected' ? 'Live (SSE)' : sseStatus === 'error' ? 'SSE error — run sse_server.py' : 'Connecting...'}
          {lastRefresh && <div style={{ marginTop:2 }}>Updated: {lastRefresh.toLocaleTimeString()}</div>}
          {liveData?.model_checkpoint && <div style={{ marginTop:2, color:'#2a2f3d', wordBreak:'break-all' }}>{liveData.model_checkpoint}</div>}
        </div>
      </div>

      {/* Right: Main content */}
      <div style={{ flex:1, display:'flex', flexDirection:'column', gap:14, overflow:'auto' }}>
        {/* Signal status strip */}
        <div style={{ display:'flex', gap:8, flexWrap:'wrap' }}>
          {signals.map(({ symbol, signal }) => (
            <div key={symbol} style={{
              padding:'5px 12px', borderRadius:5, fontFamily:'IBM Plex Mono,monospace', fontSize:11,
              border: signal ? `1px solid ${signal.dir==='long'?'rgba(34,197,94,0.4)':'rgba(239,68,68,0.4)'}` : '1px solid #1e2230',
              background: signal ? (signal.dir==='long'?'rgba(34,197,94,0.08)':'rgba(239,68,68,0.08)') : '#111318',
              color: signal ? (signal.dir==='long'?'#22c55e':'#ef4444') : '#3a3f52',
            }}>
              <span style={{ fontWeight:700 }}>{symbol.replace('USDT','')}</span>
              {signal
                ? <span style={{ marginLeft:8 }}>{signal.dir.toUpperCase()} · {signal.horizon}h · {signal.threshold}% · {((signal.prob||0)*100).toFixed(0)}%</span>
                : <span style={{ marginLeft:8, color:'#2a2f3d' }}>NO SIGNAL</span>
              }
            </div>
          ))}
          {!liveData && (
            <div style={{ padding:'5px 12px', borderRadius:5, border:'1px solid #1e2230', color:'#505870', fontFamily:'IBM Plex Mono,monospace', fontSize:11 }}>
              No live data — run: python -m python.live_predict --loop
            </div>
          )}
        </div>

        {/* OHLCV + Fan chart */}
        <div style={{ padding:0, overflow:'hidden' }}>
          <OHLCVChart symbol={selected} interval="1h" height={340} fanData={fanData} showVolume={true} showIntervalPicker={true} />
        </div>

        {/* Bottom row */}
        <div style={{ display:'grid', gridTemplateColumns:'auto 1fr', gap:14 }}>
          <div style={{ ...cardS, minWidth:120 }}>
            <div style={titleS}>Horizon</div>
            <div style={{ display:'flex', flexDirection:'column', gap:6 }}>
              {HORIZONS_H.map(h => (
                <button key={h} onClick={() => setSelectedHorizon(h)} style={{
                  padding:'6px 14px', fontFamily:'IBM Plex Mono,monospace', fontSize:11,
                  background: selectedHorizon===h ? 'rgba(59,130,246,0.15)' : 'transparent',
                  color: selectedHorizon===h ? '#3b82f6' : '#505870',
                  border: '1px solid ' + (selectedHorizon===h ? '#3b82f6' : '#1e2230'),
                  borderRadius:4, cursor:'pointer', textAlign:'left',
                }}>{h}h</button>
              ))}
            </div>
          </div>

          <div style={cardS}>
            <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:10 }}>
              <span style={titleS}>Threshold Probabilities — {selected.replace('USDT','')} {selectedHorizon}h</span>
              {selHorizonData && (
                <div style={{ display:'flex', gap:16 }}>
                  {['long','short'].map(dir => {
                    const p50 = selHorizonData?.[dir]?.quantiles?.p50;
                    return (
                      <span key={dir} style={{ fontSize:10, fontFamily:'IBM Plex Mono,monospace', color: dir==='long'?'#22c55e':'#ef4444' }}>
                        {dir} p50: {p50}%
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
            <div style={{ height:180 }}>
              <ProbabilityChart horizonData={selHorizonData} entryProbLine={bestCombo?.entry_prob} />
            </div>
          </div>
        </div>

        {/* Quantile table */}
        {selPred && (
          <div style={cardS}>
            <div style={titleS}>Quantile Predictions — {selected.replace('USDT','')}</div>
            <div style={{ overflowX:'auto' }}>
              <table style={{ borderCollapse:'collapse', width:'100%' }}>
                <thead>
                  <tr style={{ background:'#0f1014' }}>
                    {['Horizon','Dir','p10','p25','p50','p75','p90'].map(h => (
                      <th key={h} style={{ padding:'5px 10px', textAlign:'right', fontSize:10, color:'#505870', fontFamily:'IBM Plex Mono,monospace', fontWeight:400, borderBottom:'1px solid #1e2230', whiteSpace:'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {selPred.horizons.flatMap(h => ['long','short'].map(dir => (
                    <tr key={`${h.horizon_h}-${dir}`} style={{ borderBottom:'1px solid #181b26' }}>
                      <td style={{ padding:'4px 10px', textAlign:'right', fontSize:11, color:'#8890a4', fontFamily:'IBM Plex Mono,monospace' }}>{h.horizon_h}h</td>
                      <td style={{ padding:'4px 10px', textAlign:'right', fontSize:11, fontFamily:'IBM Plex Mono,monospace', color: dir==='long'?'#22c55e':'#ef4444' }}>{dir}</td>
                      {['p10','p25','p50','p75','p90'].map(q => (
                        <td key={q} style={{ padding:'4px 10px', textAlign:'right', fontSize:11, fontFamily:'IBM Plex Mono,monospace', color:'#c5c9d6' }}>
                          {dir==='long'?'+':'-'}{h[dir]?.quantiles?.[q] || '—'}%
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
};

Object.assign(window, { LiveSection });
