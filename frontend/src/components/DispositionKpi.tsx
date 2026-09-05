import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts'
import { decisionDistribution } from '../data/mockData'
import type { DecisionDistribution } from '../data/mockData'

const COLOR: Record<string, string> = {
  Investigate: '#e5484d',
  Remediate: '#e3a008',
  Monitor: '#3aa663',
  Clear: '#8494ad',
}

export default function DispositionKpi() {
  const data: DecisionDistribution[] = Array.isArray(decisionDistribution)
    ? decisionDistribution
    : (decisionDistribution as any)()

  const total = data.reduce((sum, d) => sum + d.count, 0)

  return (
    <div className="rounded-2xl border border-gold/30 bg-parchment-deep/60 p-4 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-serif text-lg text-navy">Decision Distribution</h3>
        <span className="text-[11px] text-navy/50">20,000 Portfolio</span>
      </div>
      <div className="flex items-center gap-4">
        <div className="relative h-[160px] w-[160px] shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                dataKey="count"
                nameKey="decision"
                innerRadius={46}
                outerRadius={70}
                paddingAngle={3}
              >
                {data.map((d) => (
                  <Cell key={d.decision} fill={COLOR[d.decision] || '#8884d8'} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
            <span className="font-serif text-xl font-bold text-navy">
              {total.toLocaleString('en-IN')}
            </span>
            <span className="text-[10px] uppercase tracking-wider text-navy/60">Works</span>
          </div>
        </div>
        <ul className="flex-1 space-y-2 text-xs">
          {data.map((d) => (
            <li key={d.decision} className="flex items-center justify-between">
              <span className="flex items-center gap-2 font-medium text-navy">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: COLOR[d.decision] || '#8884d8' }}
                />
                {d.decision}
              </span>
              <span className="font-mono text-navy/70">
                {d.count.toLocaleString('en-IN')}{' '}
                <span className="text-[10px] text-navy/40">
                  ({total > 0 ? ((d.count / total) * 100).toFixed(1) : 0}%)
                </span>
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
