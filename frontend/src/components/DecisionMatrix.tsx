import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'
import { riskConfidencePoints } from '../data/mockData'
import type { RiskConfidencePoint } from '../data/mockData'

const QUADRANT_COLOR: Record<string, string> = {
  Investigate: '#e5484d',
  Remediate: '#e3a008',
  Monitor: '#3aa663',
  Clear: '#8494ad',
}

export default function DecisionMatrix() {
  const points: RiskConfidencePoint[] = riskConfidencePoints()

  const byDecision = points.reduce<Record<string, RiskConfidencePoint[]>>((acc, p) => {
    ;(acc[p.decision] ??= []).push(p)
    return acc
  }, {})

  return (
    <div className="rounded-2xl border border-gold/30 bg-parchment-deep/60 p-4 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-serif text-lg text-navy">Risk vs Confidence Matrix</h3>
        <span className="text-[11px] text-navy/50">2×2 Decision Policy</span>
      </div>
      <div className="h-[250px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#0d1b2a15" />
            <XAxis
              type="number"
              dataKey="risk"
              name="Risk Score (R)"
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: '#0d1b2a99' }}
              label={{ value: 'Substantive Risk (R) →', position: 'insideBottom', offset: -10, fontSize: 10, fill: '#0d1b2a80' }}
            />
            <YAxis
              type="number"
              dataKey="confidence"
              name="Confidence (C)"
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: '#0d1b2a99' }}
              label={{ value: 'Evidentiary Confidence (C) →', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#0d1b2a80' }}
            />
            <ZAxis range={[28, 28]} />
            <ReferenceLine x={0.5} stroke="#0d1b2a40" strokeDasharray="4 4" />
            <ReferenceLine y={0.5} stroke="#0d1b2a40" strokeDasharray="4 4" />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              formatter={(value: any) => typeof value === 'number' ? value.toFixed(2) : value}
              contentStyle={{ backgroundColor: '#f5efe0', borderColor: '#c9a227', borderRadius: '8px', fontSize: '11px' }}
            />
            {Object.entries(byDecision).map(([decision, pts]) => (
              <Scatter key={decision} name={decision} data={pts} fill={QUADRANT_COLOR[decision] || '#8884d8'} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 rounded-lg border border-navy/10 bg-white/40 p-2 text-[10px] text-navy/70">
        <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-[#3aa663]" />↖ Monitor (Low R, High C)</span>
        <span className="flex items-center justify-end gap-1">Investigate (High R, High C) ↗<span className="h-1.5 w-1.5 rounded-full bg-[#e5484d]" /></span>
        <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-[#8494ad]" />↙ Clear / Backlog</span>
        <span className="flex items-center justify-end gap-1">Remediate / Audit (High R, Low C) ↘<span className="h-1.5 w-1.5 rounded-full bg-[#e3a008]" /></span>
      </div>
    </div>
  )
}
