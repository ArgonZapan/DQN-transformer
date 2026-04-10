import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7c7c', '#00C49F']

export default function EquityChart({ actors }) {
  if (!actors || Object.keys(actors).length === 0) {
    return <div className="no-data">Brak danych od aktorów</div>
  }

  const actorKeys = Object.keys(actors)
  
  // Build separate data arrays per actor
  const seriesData = {}
  for (const key of actorKeys) {
    const entries = actors[key] || []
    if (entries.length === 0) continue
    
    const symbol = key.replace('actor:', '')
    seriesData[symbol] = entries.map((entry, i) => ({
      index: i,
      [symbol]: entry.cumulative_pnl ?? entry.episode_pnl ?? entry.totalSteps ?? 0,
      time: new Date(entry.timestamp).toLocaleTimeString()
    }))
  }
  
  const symbols = Object.keys(seriesData)
  
  if (symbols.length === 0) {
    return <div className="no-data">Aktorzy jeszcze nie wysłali metryk</div>
  }
  
  // Merge all series into one array for the chart
  const maxLength = Math.max(...symbols.map(s => seriesData[s].length))
  const chartData = []
  
  for (let i = 0; i < maxLength; i++) {
    const point = { index: i }
    for (const symbol of symbols) {
      const entry = seriesData[symbol][i]
      if (entry) {
        point[symbol] = entry[symbol]
        point.time = entry.time
      }
    }
    chartData.push(point)
  }

  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2d333b" />
        <XAxis 
          dataKey="index" 
          stroke="#484f58" 
          fontSize={11} 
          tickFormatter={(idx) => chartData[idx]?.time || ''} 
        />
        <YAxis stroke="#484f58" fontSize={11} />
        <Tooltip
          contentStyle={{ background: '#161b22', border: '1px solid #2d333b', borderRadius: 6 }}
          labelStyle={{ color: '#8b949e' }}
          formatter={(value, name) => [value?.toFixed(4) || 0, name]}
        />
        <Legend />
        {symbols.map((symbol, i) => (
          <Line
            key={symbol}
            name={symbol}
            type="monotone"
            dataKey={symbol}
            stroke={COLORS[i % COLORS.length]}
            dot={false}
            strokeWidth={2}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
