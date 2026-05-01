import { useState } from 'react';
import BacktestSection from '../components/BacktestSection.jsx';
import StrategySection from '../components/StrategySection.jsx';
import LiveSection from '../components/LiveSection.jsx';

const TABS = [
  { id: 'backtest', label: 'Backtest Explorer' },
  { id: 'strategy', label: 'Strategy Simulator' },
  { id: 'live',     label: 'Live Predictions' },
];

export default function App() {
  const [tab, setTab] = useState('backtest');
  const [bestCombo, setBestCombo] = useState(null);

  const tabBtn = (id) => ({
    padding: '12px 20px',
    fontFamily: 'IBM Plex Mono, monospace',
    fontSize: 11,
    fontWeight: tab === id ? 600 : 400,
    color: tab === id ? '#e2e4ea' : '#505870',
    background: 'transparent',
    border: 'none',
    borderBottom: tab === id ? '2px solid #3b82f6' : '2px solid transparent',
    cursor: 'pointer',
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
    transition: 'color 0.15s',
    marginBottom: '-1px',
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <div style={{ background: '#0b0c0f', borderBottom: '1px solid #1e2230', padding: '10px 20px', display: 'flex', alignItems: 'center', gap: 12 }}>
        <span style={{ fontWeight: 700, color: '#e2e4ea', fontSize: 14, letterSpacing: '0.06em' }}>QUANTILENET</span>
        <span style={{ color: '#3a3f52', fontSize: 10 }}>TFT Price Forecasting</span>
        <div style={{ marginLeft: 'auto', fontSize: 10, color: '#3a3f52' }}>
          {bestCombo && (
            <span>
              Best: {bestCombo.symbol} {bestCombo.horizon}h {bestCombo.threshold}% &nbsp;
              <span style={{ color: '#22c55e' }}>SQN {bestCombo.sqn}</span>
            </span>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 0, borderBottom: '1px solid #1e2230', background: '#0b0c0f', padding: '0 20px' }}>
        {TABS.map(t => (
          <button key={t.id} style={tabBtn(t.id)} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </div>

      <div style={{ flex: 1, overflow: 'auto', padding: 20 }}>
        {tab === 'backtest' && <BacktestSection />}
        {tab === 'strategy' && <StrategySection onBestCombo={setBestCombo} />}
        {tab === 'live'     && <LiveSection bestCombo={bestCombo} />}
      </div>
    </div>
  );
}
