import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function EpsilonChart({ data }) {
  if (!data || data.length === 0) {
    return <div className="no-data">Brak danych</div>
  }

  const chartData = data
    .filter(d => d.epsilon !== undefined)
    .map(d => ({
      time: new Date(d.timestamp).toLocaleTimeString(),
      epsilon: typeof d.epsilon === 'number' ? parseFloat(d.epsilon.toFixed(4)) : 0,
      step: d.step || 0
    }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2d333b" />
        <XAxis dataKey="time" stroke="#484f58" fontSize={11} />
        <YAxis domain={[0, 1]} stroke="#484f58" fontSize={11} />
        <Tooltip
          contentStyle={{ background: '#161b22', border: '1px solid #2d333b', borderRadius: 6 }}
          labelStyle={{ color: '#8b949e' }}
        />
        <Line type="monotone" dataKey="epsilon" stroke="#82ca9d" dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  )
}
