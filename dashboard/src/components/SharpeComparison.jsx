import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

export default function SharpeComparison({ data }) {
  if (!data || data.length === 0) {
    return <div className="no-data">Brak danych Sharpe ratio</div>
  }

  const latest = data[data.length - 1]
  const inSample = latest.sharpe_in_sample || 0
  const outOfSample = latest.sharpe_out_of_sample || 0

  const chartData = [
    { period: 'Current', in_sample: parseFloat(inSample.toFixed(3)), out_of_sample: parseFloat(outOfSample.toFixed(3)) }
  ]

  if (inSample === 0 && outOfSample === 0) {
    return (
      <div className="no-data">
        Sharpe ratio będzie dostępne po ewaluacji out-of-sample
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2d333b" />
        <XAxis dataKey="period" stroke="#484f58" fontSize={11} />
        <YAxis stroke="#484f58" fontSize={11} />
        <Tooltip
          contentStyle={{ background: '#161b22', border: '1px solid #2d333b', borderRadius: 6 }}
          labelStyle={{ color: '#8b949e' }}
        />
        <Legend />
        <Bar dataKey="in_sample" fill="#8884d8" name="In-Sample" />
        <Bar dataKey="out_of_sample" fill="#82ca9d" name="Out-of-Sample" />
      </BarChart>
    </ResponsiveContainer>
  )
}
