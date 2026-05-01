import { useEffect, useRef, useState } from 'react';
import { createChart, CrosshairMode, LineStyle } from 'lightweight-charts';

const TF_SECONDS = { '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400 };
const INTERVALS = ['1m', '5m', '15m', '1h', '4h', '1d'];
const PROGRESS_THRESHOLD = 1000;     // show % only when fetching > this many candles

// Single call to /api/klines — backend handles pagination, retries, rate limits.
async function fetchAllKlines(symbol, tf, maxCandles, onProgress) {
  onProgress?.(10);
  const url = `/api/klines?symbol=${symbol}&interval=${tf}&limit=${maxCandles}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error('klines ' + r.status);
  const data = await r.json();
  onProgress?.(100);
  return data;
}

const QUANTILES = [
  ['p90', 3, LineStyle.Dotted, 0.90],
  ['p75', 3, LineStyle.Dotted, 0.90],
  ['p50', 3, LineStyle.Solid,  0.90],
  ['p25', 3, LineStyle.Dotted, 0.90],
  ['p10', 3, LineStyle.Dotted, 0.90],
];

const ALL_QUANTILE_KEYS = ['p10', 'p25', 'p50', 'p75', 'p90'];

export default function OHLCVChart({
  symbol = 'BTCUSDT',
  interval: initialInterval = '15m',
  height = 380,
  markers = [],
  priceLines = [],
  fanData = null,
  showVolume = true,
  showIntervalPicker = true,
  onLastPrice = null,
  maxCandles = 500,
}) {
  const containerRef = useRef(null);
  const chartRef     = useRef(null);
  const candleSeriesRef = useRef(null);
  const candleBaseRef   = useRef(null);
  const candlesRef      = useRef([]);   // all fetched candles, for historical fans
  const fanSeriesRef    = useRef([]);
  const priceLineRefs   = useRef([]);

  const [tf, setTf] = useState(initialInterval);
  const [loading, setLoading] = useState(true);
  const [loadingPct, setLoadingPct] = useState(0);
  const [error, setError] = useState(null);
  const [lastPrice, setLastPrice] = useState(null);
  const [priceDir, setPriceDir] = useState(1);
  const [candleReady, setCandleReady] = useState(0);

  useEffect(() => { setTf(initialInterval); }, [initialInterval]);

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    candleBaseRef.current = null;
    candlesRef.current = [];
    fanSeriesRef.current = [];

    const chart = createChart(el, {
      layout: { background: { color: 'transparent' }, textColor: '#8890a4' },
      grid: { vertLines: { color: '#181b26' }, horzLines: { color: '#181b26' } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#1e2230' },
      timeScale: { borderColor: '#1e2230', timeVisible: true, secondsVisible: false },
      width: el.clientWidth,
      height,
    });
    chartRef.current = chart;

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e', downColor: '#ef4444',
      borderUpColor: '#22c55e', borderDownColor: '#ef4444',
      wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    });
    candleSeriesRef.current = candleSeries;

    let volSeries = null;
    if (showVolume) {
      volSeries = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'vol' });
      chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
    }

    const ro = new ResizeObserver(() => {
      if (chartRef.current) chartRef.current.applyOptions({ width: el.clientWidth });
    });
    ro.observe(el);

    let cancelled = false;
    (async () => {
      setLoading(true); setLoadingPct(0); setError(null);
      try {
        const raw = await fetchAllKlines(symbol, tf, maxCandles, pct => {
          if (!cancelled) setLoadingPct(pct);
        });
        if (cancelled) return;

        const candles = raw.map(k => ({
          time: Math.floor(k[0] / 1000),
          open: +k[1], high: +k[2], low: +k[3], close: +k[4], volume: +k[5],
        }));
        candlesRef.current = candles;

        const tfSecs = TF_SECONDS[tf] || 3600;
        const futureHours = fanData ? Math.max(8, ...fanData.horizons.map(h => h.h)) : 4;
        const futureBars = Math.ceil((futureHours * 3600) / tfSecs);
        const lastT = candles[candles.length - 1].time;
        const whitespace = Array.from({ length: futureBars }, (_, i) => ({
          time: lastT + (i + 1) * tfSecs,
        }));
        candleSeries.setData([...candles, ...whitespace]);

        if (volSeries) {
          volSeries.setData(candles.map(c => ({
            time: c.time, value: c.volume,
            color: c.close >= c.open ? 'rgba(34,197,94,0.22)' : 'rgba(239,68,68,0.22)',
          })));
        }

        if (markers?.length) {
          const sorted = [...markers].filter(m => m.time_ms).sort((a, b) => a.time_ms - b.time_ms);
          candleSeries.setMarkers(sorted.map(m => ({
            time: Math.floor(m.time_ms / 1000),
            position: m.position ?? (m.type === 'sl' ? 'aboveBar' : 'belowBar'),
            color: m.color ?? (m.type === 'tp' ? '#22c55e' : m.type === 'sl' ? '#ef4444' : '#6b7280'),
            shape: m.shape ?? (m.type === 'tp' ? 'arrowUp' : m.type === 'sl' ? 'arrowDown' : 'circle'),
            text: m.text ?? (m.type ? m.type.toUpperCase() : ''),
            size: m.size ?? 1,
          })));
        }

        // priceLines applied via dedicated effect below.

        const lc = candles[candles.length - 1];
        const prev = candles[candles.length - 2];
        setLastPrice(lc.close);
        setPriceDir(lc.close >= (prev?.close ?? lc.close) ? 1 : -1);
        onLastPrice?.(lc.close);

        const RECENT = 120;
        const fromIdx = Math.max(0, candles.length - RECENT);
        chart.timeScale().setVisibleRange({
          from: candles[fromIdx].time,
          to: lastT + futureBars * tfSecs,
        });

        candleBaseRef.current = { time: lc.time, close: lc.close };
        setCandleReady(n => n + 1);
        setLoading(false);
      } catch (e) {
        if (!cancelled) { setError(e.message); setLoading(false); }
      }
    })();

    return () => {
      cancelled = true;
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
    };
  }, [symbol, tf, height, showVolume, maxCandles]);

  // Fan overlay: current prediction + historical verification fans
  useEffect(() => {
    const chart = chartRef.current;
    const base  = candleBaseRef.current;
    if (!chart || !fanData || !base) return;

    fanSeriesRef.current.forEach(s => { try { chart.removeSeries(s); } catch { /* noop */ } });
    fanSeriesRef.current = [];

    const sel = fanData.selectedHorizon ?? null;  // null = all horizons
    const allHorizons = (fanData.horizons || []).filter(h => h.long && h.short);
    const horizons = sel ? allHorizons.filter(h => h.h === sel) : allHorizons;
    if (!horizons.length) return;

    const allowedQ = Array.isArray(fanData.enabledQuantiles)
      ? fanData.enabledQuantiles
      : ALL_QUANTILE_KEYS;
    const activeQuantiles = QUANTILES.filter(([q]) => allowedQ.includes(q));

    const addLine = (color, width, style, t0, t1, value) => {
      if (t1 <= t0) return;
      const s = chart.addLineSeries({
        color, lineWidth: width, lineStyle: style,
        priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
      });
      s.setData([{ time: t0, value }, { time: t1, value }]);
      fanSeriesRef.current.push(s);
    };

    // ── Current (future) fan ─────────────────────────────────────────────────
    const baseTime  = base.time;
    const basePrice = base.close;

    const maxHorizonEnd = baseTime + Math.max(...horizons.map(h => h.h)) * 3600;
    addLine('rgba(255,255,255,0.55)', 1, LineStyle.Solid, baseTime, maxHorizonEnd, basePrice);

    horizons.forEach(h => {
      const tEnd = baseTime + h.h * 3600;
      activeQuantiles.forEach(([q, w, st, alpha]) => {
        const extreme = q === 'p10' || q === 'p90';
        const longColor  = extreme ? `rgba(134,239,172,${alpha})` : `rgba(34,197,94,${alpha})`;
        const shortColor = extreme ? `rgba(251,146,60,${alpha})`  : `rgba(239,68,68,${alpha})`;
        const lw = 3;
        addLine(longColor,  lw, st, baseTime, tEnd, basePrice * (1 + (h.long[q]  ?? 0) / 100));
        addLine(shortColor, lw, st, baseTime, tEnd, basePrice * (1 - (h.short[q] ?? 0) / 100));
      });
    });

    // ── Historical verification fans (real model inference per anchor) ────────
    const histFans = fanData.historicalFans;
    if (histFans?.length) {
      histFans.forEach((fan, idx) => {
        const aTime  = fan.anchorTime;
        const aPrice = fan.anchorPrice;
        const allAH  = (fan.horizons || []).filter(h => h.long && h.short);
        const aHorizons = sel ? allAH.filter(h => h.h === sel) : allAH;
        if (!aTime || !aPrice || !aHorizons.length) return;

        const maxAEnd = Math.min(aTime + Math.max(...aHorizons.map(h => h.h)) * 3600, baseTime);
        if (maxAEnd > aTime) {
          addLine('rgba(255,255,255,0.55)', 1, LineStyle.Solid, aTime, maxAEnd, aPrice);
        }

        aHorizons.forEach(h => {
          const tEnd = Math.min(aTime + h.h * 3600, baseTime);
          if (tEnd <= aTime) return;
          activeQuantiles.forEach(([q, w, , alpha]) => {
            const a = (alpha * 0.5).toFixed(3);
            const extreme = q === 'p10' || q === 'p90';
            const longColor  = extreme ? `rgba(134,239,172,${a})` : `rgba(34,197,94,${a})`;
            const shortColor = extreme ? `rgba(251,146,60,${a})`  : `rgba(239,68,68,${a})`;
            const lw = 3;
            addLine(longColor,  lw, LineStyle.Dashed,
              aTime, tEnd, aPrice * (1 + (h.long[q]  ?? 0) / 100));
            addLine(shortColor, lw, LineStyle.Dashed,
              aTime, tEnd, aPrice * (1 - (h.short[q] ?? 0) / 100));
          });
        });
      });
    }
  }, [fanData, candleReady, tf]);

  // Reactive price lines (EP / TP / SL).
  useEffect(() => {
    const series = candleSeriesRef.current;
    if (!series) return;
    priceLineRefs.current.forEach(pl => { try { series.removePriceLine(pl); } catch { /* noop */ } });
    priceLineRefs.current = [];
    (priceLines || []).forEach(pl => {
      const line = series.createPriceLine({
        price: pl.price,
        color: pl.color || '#3b82f6',
        lineStyle: LineStyle.Dashed,
        lineWidth: 1,
        title: pl.title || '',
        axisLabelVisible: true,
      });
      priceLineRefs.current.push(line);
    });
  }, [priceLines, candleReady]);

  const fmt = p => p
    ? (p > 1000 ? p.toLocaleString('en-US', { maximumFractionDigits: 2 }) : p.toFixed(4))
    : '—';

  return (
    <div style={{ position: 'relative', background: '#0c0d11', borderRadius: 6, overflow: 'hidden', border: '1px solid #1e2230' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 12px', borderBottom: '1px solid #1e2230', background: '#0b0c0f' }}>
        <span style={{ fontFamily: 'IBM Plex Mono,monospace', fontWeight: 700, color: '#e2e4ea', fontSize: 13 }}>{symbol}</span>
        {lastPrice != null && (
          <span style={{ fontFamily: 'IBM Plex Mono,monospace', color: priceDir >= 0 ? '#22c55e' : '#ef4444', fontSize: 12 }}>
            {fmt(lastPrice)}
          </span>
        )}
        {!loading && (
          <span style={{ fontFamily: 'IBM Plex Mono,monospace', color: '#3a3f52', fontSize: 10, marginLeft: 4 }}>
            {maxCandles.toLocaleString()} candles
          </span>
        )}
        {showIntervalPicker && (
          <div style={{ display: 'flex', gap: 3, marginLeft: 'auto' }}>
            {INTERVALS.map(iv => (
              <button key={iv} onClick={() => setTf(iv)} style={{
                padding: '2px 7px', fontSize: 10, fontFamily: 'IBM Plex Mono,monospace',
                background: tf === iv ? '#3b82f6' : 'transparent',
                color: tf === iv ? '#fff' : '#505870',
                border: '1px solid ' + (tf === iv ? '#3b82f6' : '#1e2230'),
                borderRadius: 3, cursor: 'pointer', transition: 'all 0.15s',
              }}>{iv}</button>
            ))}
          </div>
        )}
      </div>
      <div ref={containerRef} style={{ width: '100%', height }} />
      {loading && (
        <div style={{ position: 'absolute', inset: 0, top: 33, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, background: 'rgba(11,12,15,0.82)', color: '#8890a4', fontSize: 12, fontFamily: 'IBM Plex Mono,monospace' }}>
          <span>Loading {symbol}…</span>
          {maxCandles > PROGRESS_THRESHOLD && (
            <div style={{ width: 160, height: 3, background: '#1e2230', borderRadius: 2 }}>
              <div style={{ width: `${loadingPct}%`, height: '100%', background: '#3b82f6', borderRadius: 2, transition: 'width 0.2s' }} />
            </div>
          )}
          {maxCandles > PROGRESS_THRESHOLD && (
            <span style={{ fontSize: 10, color: '#3a3f52' }}>{loadingPct}%</span>
          )}
        </div>
      )}
      {error && (
        <div style={{ position: 'absolute', inset: 0, top: 33, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 6, background: 'rgba(11,12,15,0.9)', color: '#ef4444', fontSize: 12 }}>
          <span>&#9888; {error}</span>
          <span style={{ color: '#505870', fontSize: 11 }}>Check CORS / network</span>
        </div>
      )}
    </div>
  );
}
