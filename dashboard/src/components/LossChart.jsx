import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function LossChart({ data }) {
  if (!data || data.length === 0) {
    return <div className="no-data">Brak danych — Learner jeszcze nie wysłał metryk</div>
  }

  const chartData = data
    .filter(d => d.loss !== undefined)
    .map(d => ({
      time: new Date(d.timestamp).toLocaleTimeString(),
      loss: typeof d.loss === 'number' ? parseFloat(d.loss.toFixed(6)) : 0,
      step: d.step || 0
    }))

  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2d333b" />
        <XAxis dataKey="time" stroke="#484f58" fontSize={11} />
        <YAxis stroke="#484f58" fontSize={11} />
        <Tooltip
          contentStyle={{ background: '#161b22', border: '1px solid #2d333b', borderRadius: 6 }}
          labelStyle={{ color: '#8b949e' }}
        />
        <Line type="monotone" dataKey="loss" stroke="#8884d8" dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  )
}
