export default function BufferMetrics({ data }) {
  if (!data || data.length === 0) {
    return <div className="no-data">Brak danych</div>
  }

  const latest = data[data.length - 1]
  const bufferSize = latest.buffer_size || 0
  const capacity = 500000
  const fillPercent = Math.min(100, (bufferSize / capacity) * 100)
  const step = latest.step || 0

  return (
    <div>
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="value">{bufferSize.toLocaleString()}</div>
          <div className="label">Buffer Size</div>
        </div>
        <div className="metric-card">
          <div className="value">{step.toLocaleString()}</div>
          <div className="label">Train Steps</div>
        </div>
      </div>

      <div style={{ marginTop: 12 }}>
        <div style={{ fontSize: '0.8rem', color: '#8b949e', marginBottom: 4 }}>
          Buffer Fill: {fillPercent.toFixed(1)}%
        </div>
        <div className="buffer-bar">
          <div className="buffer-fill" style={{ width: `${fillPercent}%` }}>
            {fillPercent > 10 ? `${fillPercent.toFixed(0)}%` : ''}
          </div>
        </div>
      </div>

      <div className="metrics-grid" style={{ marginTop: 12 }}>
        <div className="metric-card">
          <div className="value" style={{ fontSize: '1.1rem' }}>
            {latest.epsilon !== undefined ? latest.epsilon.toFixed(4) : '—'}
          </div>
          <div className="label">Epsilon</div>
        </div>
        <div className="metric-card">
          <div className="value" style={{ fontSize: '1.1rem' }}>
            {latest.loss !== undefined ? latest.loss.toFixed(6) : '—'}
          </div>
          <div className="label">Last Loss</div>
        </div>
      </div>
    </div>
  )
}
