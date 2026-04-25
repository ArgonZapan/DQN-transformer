// OHLCVChart — TradingView Lightweight Charts wrapper
// Fetches OHLCV from Binance REST API, supports markers, price lines, fan overlays

const OHLCVChart = ({
  symbol = 'BTCUSDT',
  interval: initialInterval = '15m',
  height = 380,
  markers = [],
  priceLines = [],
  fanData = null,
  showVolume = true,
  showIntervalPicker = true,
  onLastPrice = null,
}) => {
  const containerRef  = React.useRef(null);
  const chartRef      = React.useRef(null);
  const candleBaseRef = React.useRef(null);
  const fanSeriesRef  = React.useRef([]);

  const [tf, setTf]               = React.useState(initialInterval);
  const [loading, setLoading]     = React.useState(true);
  const [error, setError]         = React.useState(null);
  const [lastPrice, setLastPrice] = React.useState(null);
  const [priceDir, setPriceDir]   = React.useState(1);
  const [candleReady, setCandleReady] = React.useState(0);

  const INTERVALS = ['1m','5m','15m','1h','4h','1d'];

  React.useEffect(() => { setTf(initialInterval); }, [initialInterval]);

  React.useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    candleBaseRef.current = null;
    fanSeriesRef.current  = [];

    const chart = LightweightCharts.createChart(el, {
      layout: { background: { color: 'transparent' }, textColor: '#8890a4' },
      grid: { vertLines: { color: '#181b26' }, horzLines: { color: '#181b26' } },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#1e2230' },
      timeScale: { borderColor: '#1e2230', timeVisible: true, secondsVisible: false },
      width: el.clientWidth,
      height: height,
    });
    chartRef.current = chart;

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e', downColor: '#ef4444',
      borderUpColor: '#22c55e', borderDownColor: '#ef4444',
      wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    });

    let volSeries = null;
    if (showVolume) {
      volSeries = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol' });
      chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
    }

    const ro = new ResizeObserver(() => {
      if (el && chartRef.current) chartRef.current.applyOptions({ width: el.clientWidth });
    });
    ro.observe(el);

    let cancelled = false;
    const load = async () => {
      setLoading(true); setError(null);
      try {
        const r = await fetch(
          `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${tf}&limit=500`
        );
        if (!r.ok) throw new Error('Binance ' + r.status);
        if (cancelled) return;
        const raw = await r.json();
        const candles = raw.map(k => ({
          time: Math.floor(k[0] / 1000),
          open: +k[1], high: +k[2], low: +k[3], close: +k[4], volume: +k[5],
        }));

        candleSeries.setData(candles);

        if (volSeries) {
          volSeries.setData(candles.map(c => ({
            time: c.time, value: c.volume,
            color: c.close >= c.open ? 'rgba(34,197,94,0.22)' : 'rgba(239,68,68,0.22)',
          })));
        }

        if (markers && markers.length) {
          const sorted = [...markers].filter(m => m.time_ms).sort((a,b) => a.time_ms - b.time_ms);
          candleSeries.setMarkers(sorted.map(m => ({
            time: Math.floor(m.time_ms / 1000),
            position: m.type === 'sl' ? 'aboveBar' : 'belowBar',
            color: m.type === 'tp' ? '#22c55e' : m.type === 'sl' ? '#ef4444' : '#6b7280',
            shape: m.type === 'tp' ? 'arrowUp' : m.type === 'sl' ? 'arrowDown' : 'circle',
            text: m.type.toUpperCase(), size: 1,
          })));
        }

        priceLines.forEach(pl => {
          candleSeries.createPriceLine({
            price: pl.price, color: pl.color || '#3b82f6',
            lineStyle: LightweightCharts.LineStyle.Dashed, lineWidth: 1, title: pl.title || '',
          });
        });

        const lc   = candles[candles.length - 1];
        const prev = candles[candles.length - 2];
        setLastPrice(lc.close);
        setPriceDir(lc.close >= prev?.close ? 1 : -1);
        if (onLastPrice) onLastPrice(lc.close);

        const RECENT  = 120;
        const fromIdx = Math.max(0, candles.length - RECENT);
        chart.timeScale().setVisibleRange({
          from: candles[fromIdx].time,
          to:   lc.time + 3600 * 4,
        });

        candleBaseRef.current = { time: lc.time, close: lc.close };
        setCandleReady(n => n + 1);
        setLoading(false);
      } catch (e) {
        if (!cancelled) { setError(e.message); setLoading(false); }
      }
    };

    load();
    return () => {
      cancelled = true;
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [symbol, tf, height]);

  React.useEffect(() => {
    const chart = chartRef.current;
    const base  = candleBaseRef.current;
    if (!chart || !fanData || !base) return;

    fanSeriesRef.current.forEach(s => { try { chart.removeSeries(s); } catch(_) {} });
    fanSeriesRef.current = [];

    const { time: baseTime, close: basePrice } = base;
    const horizons = (fanData.horizons || []).filter(h => h.long && h.short);
    if (!horizons.length) return;

    const ptsOf = (dir, q) => [
      { time: baseTime, value: basePrice },
      ...horizons.map(h => ({
        time: baseTime + h.h * 3600,
        value: dir === 'long'
          ? basePrice * (1 + (h.long[q] ?? 0) / 100)
          : basePrice * (1 - (h.short[q] ?? 0) / 100),
      })),
    ];

    const addLine = (color, width, style, lastVis, data) => {
      const s = chart.addLineSeries({
        color, lineWidth: width, lineStyle: style,
        priceLineVisible: false, lastValueVisible: lastVis, crosshairMarkerVisible: lastVis,
      });
      s.setData(data);
      fanSeriesRef.current.push(s);
    };

    addLine('rgba(34,197,94,0.18)', 1, 2, false, ptsOf('long','p90'));
    addLine('rgba(34,197,94,0.09)', 1, 2, false, ptsOf('long','p75'));
    addLine('rgba(34,197,94,0.92)', 2, 0, true,  ptsOf('long','p50'));
    addLine('rgba(34,197,94,0.09)', 1, 2, false, ptsOf('long','p25'));
    addLine('rgba(34,197,94,0.18)', 1, 2, false, ptsOf('long','p10'));
    addLine('rgba(239,68,68,0.18)', 1, 2, false, ptsOf('short','p90'));
    addLine('rgba(239,68,68,0.09)', 1, 2, false, ptsOf('short','p75'));
    addLine('rgba(239,68,68,0.92)', 2, 0, true,  ptsOf('short','p50'));
    addLine('rgba(239,68,68,0.09)', 1, 2, false, ptsOf('short','p25'));
    addLine('rgba(239,68,68,0.18)', 1, 2, false, ptsOf('short','p10'));

    const maxH = Math.max(...horizons.map(h => h.h));
    try {
      const visible = chart.timeScale().getVisibleRange();
      if (visible) {
        chart.timeScale().setVisibleRange({ from: visible.from, to: baseTime + (maxH + 24) * 3600 });
      }
    } catch(_) {}
  }, [fanData, candleReady]);

  const fmt = p => p
    ? (p > 1000 ? p.toLocaleString('en-US', { maximumFractionDigits: 2 }) : p.toFixed(4))
    : '—';

  return (
    <div style={{ position:'relative', background:'#0c0d11', borderRadius:6, overflow:'hidden', border:'1px solid #1e2230' }}>
      <div style={{ display:'flex', alignItems:'center', gap:10, padding:'6px 12px', borderBottom:'1px solid #1e2230', background:'#0b0c0f' }}>
        <span style={{ fontFamily:'IBM Plex Mono,monospace', fontWeight:700, color:'#e2e4ea', fontSize:13 }}>{symbol}</span>
        {lastPrice != null && (
          <span style={{ fontFamily:'IBM Plex Mono,monospace', color:priceDir>=0?'#22c55e':'#ef4444', fontSize:12 }}>
            {fmt(lastPrice)}
          </span>
        )}
        {showIntervalPicker && (
          <div style={{ display:'flex', gap:3, marginLeft:'auto' }}>
            {INTERVALS.map(iv => (
              <button key={iv} onClick={() => setTf(iv)} style={{
                padding:'2px 7px', fontSize:10, fontFamily:'IBM Plex Mono,monospace',
                background: tf===iv ? '#3b82f6' : 'transparent',
                color: tf===iv ? '#fff' : '#505870',
                border:'1px solid '+(tf===iv?'#3b82f6':'#1e2230'),
                borderRadius:3, cursor:'pointer', transition:'all 0.15s',
              }}>{iv}</button>
            ))}
          </div>
        )}
      </div>
      <div ref={containerRef} style={{ width:'100%', height }} />
      {loading && (
        <div style={{ position:'absolute', inset:0, top:33, display:'flex', alignItems:'center', justifyContent:'center', background:'rgba(11,12,15,0.72)', color:'#8890a4', fontSize:12, fontFamily:'IBM Plex Mono,monospace' }}>
          Loading {symbol}...
        </div>
      )}
      {error && (
        <div style={{ position:'absolute', inset:0, top:33, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', gap:6, background:'rgba(11,12,15,0.9)', color:'#ef4444', fontSize:12 }}>
          <span>&#9888; {error}</span>
          <span style={{ color:'#505870', fontSize:11 }}>Check CORS / network</span>
        </div>
      )}
    </div>
  );
};

Object.assign(window, { OHLCVChart });
