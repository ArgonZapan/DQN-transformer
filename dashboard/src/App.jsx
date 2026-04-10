import { useState, useEffect } from 'react'
import LossChart from './components/LossChart'
import EpsilonChart from './components/EpsilonChart'
import EquityChart from './components/EquityChart'
import BufferMetrics from './components/BufferMetrics'
import SharpeComparison from './components/SharpeComparison'
import TradingMetrics from './components/TradingMetrics'
import StatusBar from './components/StatusBar'
import './App.css'

const POLL_INTERVAL = 1000

function App() {
  const [metrics, setMetrics] = useState(null)
  const [status, setStatus] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [error, setError] = useState(null)

  const fetchMetrics = async () => {
    try {
      const [metricsRes, statusRes] = await Promise.all([
        fetch('/api/metrics'),
        fetch('/api/status')
      ])
      const metricsData = await metricsRes.json()
      const statusData = await statusRes.json()
      console.log('[Dashboard] Status:', statusData)
      console.log('[Dashboard] Actors:', Object.keys(metricsData.actors || {}))
      console.log('[Dashboard] Learner metrics:', metricsData.learner?.length || 0)
      setMetrics(metricsData)
      setStatus(statusData)
      setLastUpdate(new Date())
      setError(null)
    } catch (err) {
      console.error('[Dashboard] Fetch error:', err)
      setError(`Nie można połączyć z Monitoring Service: ${err.message}`)
    }
  }

  useEffect(() => {
    fetchMetrics()
    const interval = setInterval(fetchMetrics, POLL_INTERVAL)
    return () => clearInterval(interval)
  }, [])

  const learnerData = metrics?.learner || []
  const actorsData = metrics?.actors || {}

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>🤖 Trading DQN Dashboard</h1>
        <StatusBar status={status} lastUpdate={lastUpdate} error={error} onRefresh={fetchMetrics} />
      </header>

      <main className="dashboard-grid">
        <div className="card wide">
          <h2>📉 Loss Curve</h2>
          <LossChart data={learnerData} />
        </div>

        <div className="card">
          <h2>🎲 Epsilon Decay</h2>
          <EpsilonChart data={learnerData} />
        </div>

        <div className="card">
          <h2>📊 Replay Buffer</h2>
          <BufferMetrics data={learnerData} />
        </div>

        <div className="card wide">
          <h2>💰 Equity Curves</h2>
          <EquityChart actors={actorsData} />
        </div>

        <div className="card">
          <h2>📈 Sharpe Ratio</h2>
          <SharpeComparison data={learnerData} />
        </div>

        <div className="card">
          <h2>🏆 Trading Metrics</h2>
          <TradingMetrics actors={actorsData} />
        </div>
      </main>
    </div>
  )
}

export default App
