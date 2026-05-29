import { useMemo, useState } from 'react';

const FONT = 'IBM Plex Mono, monospace';
const AXIS = '#505870';
const GRID = '#181b26';

export function ScatterChart({
  width = 480, height = 220, series = [],
  xLabel = '', yLabel = '', xMin = 0, xMax = 80, yMin = 0, yMax = 80,
  diagonal = false, connectPoints = false,
}) {
  const pad = { l: 40, r: 12, t: 10, b: 28 };
  const W = width - pad.l - pad.r;
  const H = height - pad.t - pad.b;
  const xOf = v => pad.l + ((v - xMin) / (xMax - xMin || 1)) * W;
  const yOf = v => pad.t + H - ((v - yMin) / (yMax - yMin || 1)) * H;
  const ticks = (min, max, n = 5) => Array.from({ length: n + 1 }, (_, i) => min + (max - min) * i / n);
  const xTicks = ticks(xMin, xMax);
  const yTicks = ticks(yMin, yMax);

  const [hover, setHover] = useState(null);

  return (
    <div style={{ position: 'relative', width: '100%', height }}>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
        {yTicks.map(t => (
          <g key={`y${t}`}>
            <line x1={pad.l} x2={pad.l + W} y1={yOf(t)} y2={yOf(t)} stroke={GRID} strokeWidth={1} />
            <text x={pad.l - 4} y={yOf(t) + 3} textAnchor="end" fontSize={9} fill={AXIS} fontFamily={FONT}>{t}</text>
          </g>
        ))}
        {xTicks.map(t => (
          <g key={`x${t}`}>
            <line x1={xOf(t)} x2={xOf(t)} y1={pad.t} y2={pad.t + H} stroke={GRID} strokeWidth={1} />
            <text x={xOf(t)} y={height - 10} textAnchor="middle" fontSize={9} fill={AXIS} fontFamily={FONT}>{t}</text>
          </g>
        ))}
        {diagonal && (
          <line
            x1={xOf(Math.max(xMin, yMin))} y1={yOf(Math.max(xMin, yMin))}
            x2={xOf(Math.min(xMax, yMax))} y2={yOf(Math.min(xMax, yMax))}
            stroke="rgba(255,255,255,0.2)" strokeWidth={1} strokeDasharray="4 4"
          />
        )}
        {series.map((s, si) => {
          const sorted = [...s.points].sort((a, b) => a.x - b.x);
          const path = connectPoints
            ? sorted.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xOf(p.x)},${yOf(p.y)}`).join(' ')
            : '';
          return (
            <g key={si}>
              {connectPoints && <path d={path} fill="none" stroke={s.color} strokeWidth={1.5} opacity={0.6} />}
              {sorted.map((p, i) => (
                <circle
                  key={i} cx={xOf(p.x)} cy={yOf(p.y)} r={4} fill={s.color}
                  onMouseEnter={() => setHover({ s, p })}
                  onMouseLeave={() => setHover(null)}
                  style={{ cursor: 'pointer' }}
                />
              ))}
            </g>
          );
        })}
        <text x={pad.l + W / 2} y={height - 1} textAnchor="middle" fontSize={9} fill={AXIS} fontFamily={FONT}>{xLabel}</text>
        <text x={6} y={pad.t + H / 2} textAnchor="start" fontSize={9} fill={AXIS} fontFamily={FONT} transform={`rotate(-90 6 ${pad.t + H / 2})`}>{yLabel}</text>
      </svg>
      <div style={{ position: 'absolute', top: 4, right: 8, display: 'flex', gap: 10, fontFamily: FONT, fontSize: 10 }}>
        {series.map((s, i) => (
          <span key={i} style={{ color: s.color }}>● {s.label}</span>
        ))}
      </div>
      {hover && (
        <div style={{ position: 'absolute', top: 18, left: 12, background: '#0b0c0f', border: '1px solid #1e2230', padding: '4px 8px', borderRadius: 3, fontFamily: FONT, fontSize: 10, color: '#c5c9d6', pointerEvents: 'none' }}>
          {hover.s.label} | {hover.p.tip ?? `${hover.p.x.toFixed(1)} → ${hover.p.y.toFixed(1)}`}
        </div>
      )}
    </div>
  );
}

// ── BarChart (grouped) ──────────────────────────────────────────────────────
// labels: ['p10', 'p25', ...]
// groups: [{ label, color, values: [n,n,...] }, ...]
export function BarChart({
  width = 480, height = 200,
  labels = [], groups = [],
  yMin = 0, yMax = 100, yUnit = '%',
}) {
  const pad = { l: 36, r: 10, t: 14, b: 22 };
  const W = width - pad.l - pad.r;
  const H = height - pad.t - pad.b;
  const slot = W / Math.max(1, labels.length);
  const groupCount = groups.length || 1;
  const barW = Math.min(28, (slot * 0.7) / groupCount);

  const yOf = v => pad.t + H - ((v - yMin) / (yMax - yMin || 1)) * H;
  const ticks = Array.from({ length: 5 }, (_, i) => yMin + (yMax - yMin) * i / 4);

  return (
    <div style={{ position: 'relative', width: '100%', height }}>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ width: '100%', height: '100%' }}>
        {ticks.map(t => (
          <g key={t}>
            <line x1={pad.l} x2={pad.l + W} y1={yOf(t)} y2={yOf(t)} stroke={GRID} strokeWidth={1} />
            <text x={pad.l - 4} y={yOf(t) + 3} textAnchor="end" fontSize={9} fill={AXIS} fontFamily={FONT}>{Math.round(t)}{yUnit}</text>
          </g>
        ))}
        {labels.map((lbl, i) => {
          const cx = pad.l + slot * (i + 0.5);
          return (
            <g key={lbl}>
              <text x={cx} y={height - 6} textAnchor="middle" fontSize={10} fill={AXIS} fontFamily={FONT}>{lbl}</text>
              {groups.map((g, gi) => {
                const v = g.values[i];
                if (v == null || !Number.isFinite(v)) return null;
                const y = yOf(v);
                const x = cx - (groupCount * barW) / 2 + gi * barW;
                return (
                  <rect key={gi} x={x} y={y} width={barW - 1} height={Math.max(0, yOf(yMin) - y)}
                    fill={g.color} opacity={0.8} stroke={g.border || g.color} strokeWidth={1} />
                );
              })}
            </g>
          );
        })}
      </svg>
      {groups.length > 1 && (
        <div style={{ position: 'absolute', top: 0, right: 8, display: 'flex', gap: 10, fontFamily: FONT, fontSize: 10 }}>
          {groups.map((g, i) => <span key={i} style={{ color: g.color }}>● {g.label}</span>)}
        </div>
      )}
    </div>
  );
}

// ── LineChart (e.g. equity curve) ───────────────────────────────────────────
// points: [{x, y}]; uses x/y indices for layout
export function LineChart({ width = 480, height = 180, points = [], color = '#22c55e', fill = 'rgba(34,197,94,0.08)', yUnit = '%' }) {
  const data = useMemo(() => points.filter(p => Number.isFinite(p.y)), [points]);
  const pad = { l: 40, r: 10, t: 12, b: 22 };
  const W = width - pad.l - pad.r;
  const H = height - pad.t - pad.b;
  if (!data.length) return null;
  const ys = data.map(p => p.y);
  const yMin = Math.min(0, ...ys);
  const yMax = Math.max(0, ...ys);
  const yRange = (yMax - yMin) || 1;
  const xOf = i => pad.l + (i / Math.max(1, data.length - 1)) * W;
  const yOf = v => pad.t + H - ((v - yMin) / yRange) * H;
  const path = data.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xOf(i)},${yOf(p.y)}`).join(' ');
  const area = `${path} L ${xOf(data.length - 1)},${yOf(0)} L ${xOf(0)},${yOf(0)} Z`;
  const ticks = [yMin, (yMin + yMax) / 2, yMax];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ width: '100%', height }}>
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={pad.l} x2={pad.l + W} y1={yOf(t)} y2={yOf(t)} stroke={GRID} strokeWidth={1} />
          <text x={pad.l - 4} y={yOf(t) + 3} textAnchor="end" fontSize={9} fill={AXIS} fontFamily={FONT}>{t.toFixed(1)}{yUnit}</text>
        </g>
      ))}
      <path d={area} fill={fill} />
      <path d={path} stroke={color} strokeWidth={1.5} fill="none" />
      {data.length > 1 && (
        <>
          <text x={pad.l} y={height - 6} textAnchor="start" fontSize={9} fill={AXIS} fontFamily={FONT}>{data[0].x ?? ''}</text>
          <text x={pad.l + W} y={height - 6} textAnchor="end" fontSize={9} fill={AXIS} fontFamily={FONT}>{data[data.length - 1].x ?? ''}</text>
        </>
      )}
    </svg>
  );
}

// ── Histogram ───────────────────────────────────────────────────────────────
export function Histogram({ width = 480, height = 180, values = [], bins = 12, logY = false }) {
  const stats = useMemo(() => {
    if (!values.length) return null;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const step = (max - min) / bins || 1;
    const counts = Array(bins).fill(0);
    values.forEach(v => { const bi = Math.min(bins - 1, Math.floor((v - min) / step)); counts[bi]++; });
    return { min, max, step, counts };
  }, [values, bins]);
  if (!stats) return null;
  const labels = Array.from({ length: bins }, (_, i) => (stats.min + i * stats.step).toFixed(1) + '%');
  const colors = labels.map((_, i) => (stats.min + i * stats.step) >= 0 ? 'rgba(34,197,94,0.55)' : 'rgba(239,68,68,0.55)');

  const pad = { l: 30, r: 10, t: 10, b: 22 };
  const W = width - pad.l - pad.r;
  const H = height - pad.t - pad.b;
  const slot = W / bins;
  const yMax = Math.max(...stats.counts) || 1;
  // log1p compresses tall bars while keeping zero at zero (log(1+0)=0).
  const scale = v => (logY ? Math.log1p(v) : v);
  const sMax = scale(yMax) || 1;
  const yOf = v => pad.t + H - (scale(v) / sMax) * H;
  const ticks = logY
    ? [0, 1, 10, 100, 1000].filter(t => t <= yMax).concat(yMax > 1 ? [yMax] : [])
    : [0, yMax / 2, yMax];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ width: '100%', height }}>
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={pad.l} x2={pad.l + W} y1={yOf(t)} y2={yOf(t)} stroke={GRID} strokeWidth={1} />
          <text x={pad.l - 4} y={yOf(t) + 3} textAnchor="end" fontSize={9} fill={AXIS} fontFamily={FONT}>{Math.round(t)}</text>
        </g>
      ))}
      {stats.counts.map((c, i) => {
        const x = pad.l + slot * i + 1;
        const y = yOf(c);
        return (
          <g key={i}>
            <rect x={x} y={y} width={Math.max(1, slot - 2)} height={Math.max(0, yOf(0) - y)} fill={colors[i]} />
            {(i === 0 || i === bins - 1 || i === Math.floor(bins / 2)) && (
              <text x={x + slot / 2} y={height - 6} textAnchor="middle" fontSize={9} fill={AXIS} fontFamily={FONT}>{labels[i]}</text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

// ── Donut ───────────────────────────────────────────────────────────────────
export function Donut({ width = 220, height = 180, slices = [] }) {
  const cx = 80;
  const cy = height / 2;
  const r = 60;
  const inner = 38;
  const total = slices.reduce((s, x) => s + x.value, 0) || 1;

  let acc = 0;
  const arcs = slices.map(s => {
    const a0 = (acc / total) * Math.PI * 2 - Math.PI / 2;
    acc += s.value;
    const a1 = (acc / total) * Math.PI * 2 - Math.PI / 2;
    const large = a1 - a0 > Math.PI ? 1 : 0;
    const x0 = cx + r * Math.cos(a0), y0 = cy + r * Math.sin(a0);
    const x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
    const x2 = cx + inner * Math.cos(a1), y2 = cy + inner * Math.sin(a1);
    const x3 = cx + inner * Math.cos(a0), y3 = cy + inner * Math.sin(a0);
    const d = `M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1} L ${x2} ${y2} A ${inner} ${inner} 0 ${large} 0 ${x3} ${y3} Z`;
    return { d, color: s.color, label: s.label, value: s.value };
  });

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" style={{ width: '100%', height }}>
      {arcs.map((a, i) => <path key={i} d={a.d} fill={a.color} />)}
      {arcs.map((a, i) => (
        <g key={`l${i}`} transform={`translate(${cx + r + 22} ${cy - 30 + i * 22})`}>
          <rect width={10} height={10} fill={a.color} />
          <text x={16} y={9} fontSize={11} fill="#8890a4" fontFamily={FONT}>{a.label}: {a.value}%</text>
        </g>
      ))}
    </svg>
  );
}
