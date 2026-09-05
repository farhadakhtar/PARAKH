import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts'
import { decisionDistribution } from '../data/mockData'

const COLORS: Record<string, string> = {
  Investigate: '#dc2626', // Red
  Remediate: '#ea580c',   // Orange
  Monitor: '#eab308',     // Yellow
  Clear: '#16a34a',       // Green
}

const DISPLAY_ROWS = [
  { decision: 'Investigate', count: '382', pct: '1.9%' },
  { decision: 'Remediate', count: '1,142', pct: '5.7%' },
  { decision: 'Monitor', count: '6,586', pct: '32.9%' },
  { decision: 'Clear', count: '12,890', pct: '64.5%' },
]

export default function DispositionKpi() {
  const data = decisionDistribution

  return (
    <div className="flex flex-col justify-between rounded-xl border border-[#d8cbb0] bg-[#fbf9f4] p-3.5 shadow-xs">
      <h3 className="mb-2 font-serif text-xs font-bold text-[#0b1a2d]">
        Decision Distribution
      </h3>

      <div className="flex items-center gap-3">
        {/* Donut Chart */}
        <div className="relative h-32 w-32 shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                dataKey="count"
                nameKey="decision"
                innerRadius={36}
                outerRadius={56}
                paddingAngle={2}
                startAngle={90}
                endAngle={-270}
              >
                {data.map((d) => (
                  <Cell key={d.decision} fill={COLORS[d.decision] || '#94a3b8'} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-center">
            <span className="font-serif text-sm font-bold text-[#0b1a2d]">
              20,000
            </span>
            <span className="text-[9px] font-medium text-slate-500">Works</span>
          </div>
        </div>

        {/* Legend */}
        <div className="flex-1 space-y-1.5 text-[11px]">
          {DISPLAY_ROWS.map((row) => (
            <div key={row.decision} className="flex items-center justify-between text-slate-700">
              <span className="flex items-center gap-1.5">
                <span
                  className="h-2 w-2 rounded-full shrink-0"
                  style={{ backgroundColor: COLORS[row.decision] }}
                />
                <span className="font-medium text-slate-800">{row.decision}</span>
              </span>
              <span className="font-mono text-[10.5px] text-slate-700">
                {row.count} <span className="text-slate-400">({row.pct})</span>
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
