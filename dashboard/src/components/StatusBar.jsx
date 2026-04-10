export default function StatusBar({ status, lastUpdate, error, onRefresh }) {
  const learnerConnected = status?.learnerConnected || false
  const actorsCount = status?.actorsConnected || 0

  return (
    <div className="status-bar">
      {error ? (
        <span className="error-msg">{error}</span>
      ) : (
        <>
          <span>
            <span className={`status-dot ${learnerConnected ? 'connected' : 'disconnected'}`} />
            {' '}Learner: {learnerConnected ? 'OK' : 'Offline'}
          </span>
          <span>
            <span className={`status-dot ${actorsCount > 0 ? 'connected' : 'disconnected'}`} />
            {' '}Aktorzy: {actorsCount}
          </span>
          {lastUpdate && (
            <span style={{ color: '#484f58' }}>
              Ostatni update: {lastUpdate.toLocaleTimeString()}
            </span>
          )}
        </>
      )}
      <button className="refresh-btn" onClick={onRefresh}>⟳ Odśwież</button>
    </div>
  )
}
