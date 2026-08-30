export function FeelingBar({ feelings, active, onPick }) {
  return (
    <div className="feeling-bar">
      {feelings.map((f) => (
        <button
          key={f}
          type="button"
          className={f === active ? 'active' : undefined}
          onClick={() => onPick(f)}
        >
          {f}
        </button>
      ))}
    </div>
  )
}
