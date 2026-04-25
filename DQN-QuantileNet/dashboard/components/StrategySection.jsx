// Strategy Simulator — Section 2
// Filterable/sortable table + 2x2 detail panel with OHLCV+markers, equity curve, histogram, donut
// Data: fetched from /api/strategy (served by python/server/sse_server.py)

const SYMBOLS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT'];

// ── Mini Chart.js components ─────────────────────────────────────────────────

function EquityCurveChart({ trades }) {
  const ref = React.useRef(null);
  const inst = React.useRef(null);
  React.useEffect(() => {
    if (!ref.current || !trades?.length) return;
    const ctx = ref.current.getContext('2d');
    if (inst.current) inst.current.destroy();
    let cum = 0;
    const pts = trades.map(t => { cum += t.return; return { x: new Date(t.entry_time).toLocaleDateString(), y: +cum.toFixed(2) }; });
    inst.current = new Chart(ctx, {
      type: 'line',
      data: { labels: pts.map(p=>p.x), datasets: [{
        data: pts.map(p=>p.y), borderColor: cum>=0?'#22c55e':'#ef4444',
        backgroundColor: cum>=0?'rgba(34,197,94,0.08)':'rgba(239,68,68,0.08)',
        fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2,
      }]},
      options: {
        responsive:true, maintainAspectRatio:false,
        plugins:{ legend:{display:false}, tooltip:{ callbacks:{ label: i => `${i.raw.toFixed(2)}%` } } },
        scales:{
          x:{ ticks:{color:'#505870',font:{size:8,family:'IBM Plex Mono'},maxTicksLimit:6}, grid:{color:'#181b26'} },
          y:{ ticks:{color:'#505870',font:{size:9,family:'IBM Plex Mono'}}, grid:{color:'#181b26'} },
        },
      },
    });
    return () => inst.current?.destroy();
  }, [trades]);
  return <canvas ref={ref} />;
}

function ReturnHistogram({ trades }) {
  const ref = React.useRef(null);
  const inst = React.useRef(null);
  React.useEffect(() => {
    if (!ref.current || !trades?.length) return;
    const ctx = ref.current.getContext('2d');
    if (inst.current) inst.current.destroy();
    const rets = trades.map(t=>t.return);
    const min = Math.min(...rets), max = Math.max(...rets);
    const bins = 12;
    const step = (max - min) / bins || 1;
    const counts = Array(bins).fill(0);
    rets.forEach(r => { const bi = Math.min(bins-1, Math.floor((r-min)/step)); counts[bi]++; });
    const labels = Array.from({length:bins}, (_,i) => (min+i*step).toFixed(1)+'%');
    inst.current = new Chart(ctx, {
      type:'bar',
      data:{ labels, datasets:[{
        data:counts,
        backgroundColor: labels.map((_,i) => (min+i*step) >= 0 ? 'rgba(34,197,94,0.5)' : 'rgba(239,68,68,0.5)'),
        borderWidth:0,
      }]},
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{
          x:{ticks:{color:'#505870',font:{size:8,family:'IBM Plex Mono'},maxTicksLimit:6},grid:{color:'#181b26'}},
          y:{ticks:{color:'#505870',font:{size:9,family:'IBM Plex Mono'}},grid:{color:'#181b26'}},
        },
      },
    });
    return () => inst.current?.destroy();
  }, [trades]);
  return <canvas ref={ref} />;
}

function ExitDonut({ combo }) {
  const ref = React.useRef(null);
  const inst = React.useRef(null);
  React.useEffect(() => {
    if (!ref.current || !combo) return;
    const ctx = ref.current.getContext('2d');
    if (inst.current) inst.current.destroy();
    const tp = Math.round((combo.tp_rate||0)*100);
    const sl = Math.round((combo.sl_rate||0)*100);
    const hz = 100 - tp - sl;
    inst.current = new Chart(ctx, {
      type:'doughnut',
      data:{
        labels:['TP','SL','HZ'],
        datasets:[{ data:[tp,sl,hz], backgroundColor:['#22c55e','#ef4444','#6b7280'], borderWidth:0, hoverOffset:4 }]
      },
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{
          legend:{ position:'right', labels:{color:'#8890a4',font:{size:10,family:'IBM Plex Mono'},boxWidth:10} },
          tooltip:{ callbacks:{ label: i => `${i.label}: ${i.raw}%` } },
        },
        cutout:'68%',
      },
    });
    return () => inst.current?.destroy();
  }, [combo]);
  return <canvas ref={ref} />;
}

// ── Detail Panel ─────────────────────────────────────────────────────────────
function DetailPanel({ combo, onClose }) {
  if (!combo) return null;
  const tradeMarkers = (combo.trades || []).map(t => ({ time_ms: t.entry_time, type: t.exit_type }));
  const cardS = { background:'#111318', border:'1px solid #1e2230', borderRadius:6, padding:12, display:'flex', flexDirection:'column' };
  const titleS = { fontSize:10, color:'#505870', fontFamily:'IBM Plex Mono,monospace', textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:8 };

  return (
    <div style={{ position:'fixed', inset:0, zIndex:100, display:'flex', alignItems:'flex-end', justifyContent:'flex-end', background:'rgba(0,0,0,0.6)', backdropFilter:'blur(2px)' }}
      onClick={e => { if (e.target===e.currentTarget) onClose(); }}>
      <div style={{ width:'72vw', height:'85vh', background:'#0b0c0f', borderTop:'1px solid #1e2230', borderLeft:'1px solid #1e2230', borderTopLeftRadius:10, padding:20, overflow:'auto', display:'flex', flexDirection:'column', gap:14 }}>
        <div style={{ display:'flex', alignItems:'center', gap:12, marginBottom:4 }}>
          <span style={{ fontFamily:'IBM Plex Mono,monospace', fontWeight:700, color:'#e2e4ea', fontSize:15 }}>
            {combo.symbol} · {combo.horizon}h · {combo.threshold}% · {combo.direction.toUpperCase()}
          </span>
          <span style={{ fontFamily:'IBM Plex Mono,monospace', fontSize:12, color: combo.sqn>=2?'#22c55e':combo.sqn>=1?'#f59e0b':'#ef4444' }}>
            SQN {combo.sqn}
          </span>
          <span style={{ fontFamily:'IBM Plex Mono,monospace', fontSize:12, color:'#505870' }}>
            {combo.n_trades} trades{combo.n_trades<30?' *':''}
          </span>
          <button onClick={onClose} style={{ marginLeft:'auto', background:'transparent', border:'1px solid #1e2230', color:'#8890a4', borderRadius:4, padding:'4px 12px', cursor:'pointer', fontFamily:'IBM Plex Mono,monospace', fontSize:11 }}>&#x2715; Close</button>
        </div>

        <div style={{ display:'flex', gap:2 }}>
          {[
            { label:'P(Win)', value: ((combo.p_win||0)*100).toFixed(1)+'%' },
            { label:'Avg +', value: '+'+(combo.avg_win||0)+'%' },
            { label:'Avg -', value: '-'+(combo.avg_loss||0)+'%' },
            { label:'EV', value: (combo.ev||0)+'%' },
            { label:'Sharpe', value: combo.sharpe||0 },
            { label:'Kelly', value: ((combo.kelly||0)*100).toFixed(1)+'%' },
            { label:'Max DD', value: ((combo.max_dd||0)*100).toFixed(1)+'%' },
            { label:'Total', value: (combo.total_return||0)+'%' },
          ].map(({ label, value }) => (
            <div key={label} style={{ flex:1, background:'#111318', border:'1px solid #1e2230', borderRadius:5, padding:'6px 10px' }}>
              <div style={{ fontSize:9, color:'#505870', fontFamily:'IBM Plex Mono,monospace', textTransform:'uppercase', letterSpacing:'0.08em' }}>{label}</div>
              <div style={{ fontSize:13, color:'#e2e4ea', fontFamily:'IBM Plex Mono,monospace', fontWeight:600, marginTop:2 }}>{value}</div>
            </div>
          ))}
        </div>

        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gridTemplateRows:'220px 180px', gap:12, flex:1 }}>
          <div style={cardS}>
            <div style={titleS}>OHLCV + Trade Markers</div>
            <div style={{ flex:1 }}>
              <OHLCVChart symbol={combo.symbol} interval="1d" height={165} markers={tradeMarkers} showVolume={false} showIntervalPicker={false} />
            </div>
          </div>
          <div style={cardS}>
            <div style={titleS}>Equity Curve</div>
            <div style={{ flex:1 }}><EquityCurveChart trades={combo.trades || []} /></div>
          </div>
          <div style={cardS}>
            <div style={titleS}>Return Distribution</div>
            <div style={{ flex:1 }}><ReturnHistogram trades={combo.trades || []} /></div>
          </div>
          <div style={cardS}>
            <div style={titleS}>Exit Type Breakdown</div>
            <div style={{ flex:1 }}><ExitDonut combo={combo} /></div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main Section ─────────────────────────────────────────────────────────────
const StrategySection = ({ onBestCombo }) => {
  const [combos, setCombos]   = React.useState([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError]     = React.useState(null);
  const [meta, setMeta]       = React.useState(null);
  const [runs, setRuns]       = React.useState([]);
  const [selectedRun, setSelectedRun] = React.useState('');
  const [selected, setSelected] = React.useState(null);
  const [sortKey, setSortKey] = React.useState('sqn');
  const [sortDir, setSortDir] = React.useState(-1);
  const [filters, setFilters] = React.useState({ symbol:'all', horizon:'all', threshold:'all', direction:'all', minTrades:0 });

  React.useEffect(() => {
    fetch('/api/runs').then(r=>r.json()).then(data => {
      setRuns(data);
      if (data.length > 0) setSelectedRun(data[data.length - 1]);
    }).catch(()=>{});
  }, []);

  React.useEffect(() => {
    if (!selectedRun) return;
    setLoading(true); setError(null);
    fetch(`/api/strategy/${selectedRun}`)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(data => {
        setCombos(data.combos || []);
        setMeta(data);
        if (data.combos?.length > 0 && onBestCombo) onBestCombo(data.combos[0]);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [selectedRun]);

  const handleSort = (key) => {
    if (sortKey === key) setSortDir(d => -d);
    else { setSortKey(key); setSortDir(-1); }
  };

  const filtered = combos
    .filter(c =>
      (filters.symbol==='all' || c.symbol===filters.symbol) &&
      (filters.horizon==='all' || c.horizon===+filters.horizon) &&
      (filters.threshold==='all' || c.threshold===+filters.threshold) &&
      (filters.direction==='all' || c.direction===filters.direction) &&
      c.n_trades >= filters.minTrades
    )
    .sort((a,b) => sortDir * (a[sortKey] > b[sortKey] ? 1 : -1));

  const sqnColor = (v) => v>=2?'#22c55e':v>=1?'#f59e0b':'#ef4444';

  const TH = ({ label, k }) => (
    <th onClick={() => handleSort(k)} style={{ padding:'6px 8px', textAlign:'right', color: sortKey===k?'#3b82f6':'#505870', fontSize:10, fontFamily:'IBM Plex Mono,monospace', fontWeight:400, borderBottom:'1px solid #1e2230', cursor:'pointer', whiteSpace:'nowrap', userSelect:'none', background:'#0f1014' }}>
      {label}{sortKey===k?(sortDir<0?' v':' ^'):''}
    </th>
  );

  const selectStyle = { background:'#111318', color:'#8890a4', border:'1px solid #1e2230', borderRadius:4, padding:'4px 8px', fontSize:11, fontFamily:'IBM Plex Mono,monospace' };

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:16 }}>
      {/* Run selector + meta */}
      <div style={{ display:'flex', alignItems:'center', gap:10, flexWrap:'wrap' }}>
        <span style={{ fontSize:11, color:'#505870', fontFamily:'IBM Plex Mono,monospace' }}>Run:</span>
        <select style={selectStyle} value={selectedRun} onChange={e=>setSelectedRun(e.target.value)}>
          {runs.map(r=><option key={r} value={r}>{r}</option>)}
        </select>
        {meta && (
          <>
            <span style={{ fontSize:10, color:'#3a3f52', fontFamily:'IBM Plex Mono,monospace' }}>
              {meta.test_range?.start} to {meta.test_range?.end}
            </span>
            <span style={{ fontSize:10, color:'#3a3f52', fontFamily:'IBM Plex Mono,monospace' }}>
              Entry probs: [{(meta.entry_prob_grid||[]).join(', ')}]
            </span>
          </>
        )}
      </div>

      {loading && <div style={{ color:'#505870', fontFamily:'IBM Plex Mono,monospace', fontSize:12, padding:20 }}>Loading strategy data...</div>}
      {error && <div style={{ color:'#ef4444', fontFamily:'IBM Plex Mono,monospace', fontSize:12, padding:20 }}>Error: {error} — Run a backtest first.</div>}

      {!loading && !error && (
        <>
          {/* Filters */}
          <div style={{ background:'#111318', border:'1px solid #1e2230', borderRadius:8, padding:'12px 16px', display:'flex', flexWrap:'wrap', gap:10, alignItems:'center' }}>
            <span style={{ fontSize:11, color:'#505870', fontFamily:'IBM Plex Mono,monospace', marginRight:4 }}>FILTER</span>
            <select style={selectStyle} value={filters.symbol} onChange={e=>setFilters(f=>({...f,symbol:e.target.value}))}>
              <option value="all">All Symbols</option>
              {SYMBOLS.map(s=><option key={s} value={s}>{s}</option>)}
            </select>
            <select style={selectStyle} value={filters.horizon} onChange={e=>setFilters(f=>({...f,horizon:e.target.value}))}>
              <option value="all">All Horizons</option>
              {[8,12,24,48,72].map(h=><option key={h} value={h}>{h}h</option>)}
            </select>
            <select style={selectStyle} value={filters.threshold} onChange={e=>setFilters(f=>({...f,threshold:e.target.value}))}>
              <option value="all">All Thresholds</option>
              {[1.0,1.5,2.0,3.0,5.0,8.0].map(t=><option key={t} value={t}>{t}%</option>)}
            </select>
            <select style={selectStyle} value={filters.direction} onChange={e=>setFilters(f=>({...f,direction:e.target.value}))}>
              <option value="all">All Directions</option>
              {['long','short','both'].map(d=><option key={d} value={d}>{d}</option>)}
            </select>
            <label style={{ fontSize:11, color:'#505870', fontFamily:'IBM Plex Mono,monospace', display:'flex', alignItems:'center', gap:6 }}>
              Min Trades
              <input type="number" min="0" max="200" value={filters.minTrades}
                onChange={e=>setFilters(f=>({...f,minTrades:+e.target.value}))}
                style={{ ...selectStyle, width:56 }} />
            </label>
            <span style={{ marginLeft:'auto', fontSize:11, color:'#505870', fontFamily:'IBM Plex Mono,monospace' }}>{filtered.length} combos</span>
          </div>

          {/* Table */}
          <div style={{ background:'#111318', border:'1px solid #1e2230', borderRadius:8, overflow:'hidden' }}>
            <div style={{ overflowX:'auto' }}>
              <table style={{ width:'100%', borderCollapse:'collapse', minWidth:900 }}>
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
                    const td = (content, extra={}) => (
                      <td style={{ padding:'6px 8px', textAlign:'right', fontFamily:'IBM Plex Mono,monospace', fontSize:11, borderBottom:'1px solid #181b26', color:'#8890a4', ...extra }}>{content}</td>
                    );
                    return (
                      <tr key={c.rank} onClick={() => setSelected(c)}
                        style={{ background: isSelected?'#14193a': i%2===0?'#0b0c0f':'#0e0f13', cursor:'pointer' }}
                        onMouseEnter={e => { if(!isSelected) e.currentTarget.style.background='#131520'; }}
                        onMouseLeave={e => { if(!isSelected) e.currentTarget.style.background=i%2===0?'#0b0c0f':'#0e0f13'; }}>
                        {td(c.rank, { color:'#505870' })}
                        {td(c.symbol.replace('USDT',''), { color:'#e2e4ea', fontWeight:600 })}
                        {td(c.horizon+'h')}
                        {td(c.threshold+'%')}
                        {td(c.direction, { color: c.direction==='long'?'#22c55e':c.direction==='short'?'#ef4444':'#8890a4' })}
                        {td(((c.entry_prob||0)*100).toFixed(0)+'%')}
                        {td((c.n_trades||0)+(c.n_trades<30?'*':''), { color: c.n_trades<30?'#f59e0b':'#8890a4' })}
                        {td(((c.p_win||0)*100).toFixed(1)+'%')}
                        {td('+'+(c.avg_win||0)+'%', { color:'#22c55e' })}
                        {td('-'+(c.avg_loss||0)+'%', { color:'#ef4444' })}
                        {td((c.total_return>=0?'+':'')+(c.total_return||0)+'%', { color: c.total_return>=0?'#22c55e':'#ef4444', fontWeight:600 })}
                        {td(c.sharpe||0)}
                        {td(c.sqn||0, { color:sqnColor(c.sqn||0), fontWeight:700, fontSize:12 })}
                        {td((c.ev>=0?'+':'')+(c.ev||0)+'%', { color: c.ev>=0?'#22c55e':'#ef4444' })}
                        {td(((c.kelly||0)*100).toFixed(1)+'%')}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div style={{ padding:'6px 12px', borderTop:'1px solid #1e2230', fontSize:10, color:'#3a3f52', fontFamily:'IBM Plex Mono,monospace' }}>
              * n &lt; 30 — low sample size. Click any row to inspect combo.
            </div>
          </div>
        </>
      )}

      {selected && <DetailPanel combo={selected} onClose={() => setSelected(null)} />}
    </div>
  );
};

Object.assign(window, { StrategySection });
