export default function TradingMetrics({ actors }) {
  if (!actors || Object.keys(actors).length === 0) {
    return <div className="no-data">Brak metryk tradingowych</div>
  }

  const actorKeys = Object.keys(actors)

  return (
    <div>
      {actorKeys.map(key => {
        const entries = actors[key] || []
        const latest = entries[entries.length - 1]
        if (!latest) return null

        const symbol = key.replace('actor:', '')

        return (
          <div key={key} style={{ marginBottom: 16 }}>
            <h3 style={{ fontSize: '0.85rem', color: '#58a6ff', marginBottom: 8 }}>
              {symbol}
            </h3>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="value" style={{ fontSize: '1.1rem' }}>
                  {latest.win_rate !== undefined ? `${(latest.win_rate * 100).toFixed(1)}%` : '—'}
                </div>
                <div className="label">Win Rate</div>
              </div>
              <div className="metric-card">
                <div className="value" style={{ fontSize: '1.1rem' }}>
                  {latest.profit_factor !== undefined ? latest.profit_factor.toFixed(2) : '—'}
                </div>
                <div className="label">Profit Factor</div>
              </div>
              <div className="metric-card">
                <div className="value" style={{ fontSize: '1.1rem' }}>
                  {latest.trades !== undefined ? latest.trades : latest.totalSteps !== undefined ? latest.totalSteps : '—'}
                </div>
                <div className="label">Trades / Steps</div>
              </div>
              <div className="metric-card">
                <div className="value" style={{ fontSize: '1.1rem' }}>
                  {latest.totalEpisodes !== undefined ? latest.totalEpisodes : '—'}
                </div>
                <div className="label">Episodes</div>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
