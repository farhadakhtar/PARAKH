import { topRiskDistricts, recentHighRiskWorks, riskBand, formatINR } from '../data/mockData'
import type { WorkRecord } from '../data/mockData'

const RISK_TEXT_COLOR = (r: number) =>
  r >= 0.7 ? 'text-red-600' : r >= 0.5 ? 'text-orange-500' : 'text-navy/70'

const STATUS_PILL: Record<string, string> = {
  Investigate: 'bg-red-100 text-red-700 border-red-300',
  Remediate: 'bg-amber-100 text-amber-700 border-amber-300',
  Monitor: 'bg-yellow-100 text-yellow-700 border-yellow-300',
}

export function TopRiskDistrictsTable() {
  const rows: any[] = Array.isArray(topRiskDistricts)
    ? topRiskDistricts
    : (topRiskDistricts as any)()

  return (
    <div className="rounded-2xl border border-gold/30 bg-parchment-deep/60 p-4 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-serif text-lg text-navy">Top Risk Districts</h3>
        <a href="#districts" className="text-xs font-medium text-navy/60 hover:text-navy">
          View All →
        </a>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-navy/10 text-navy/50">
              <th className="py-1.5 font-medium">#</th>
              <th className="py-1.5 font-medium">District</th>
              <th className="py-1.5 font-medium">State</th>
              <th className="py-1.5 font-medium">Risk (R)</th>
              <th className="py-1.5 font-medium">Conf (C)</th>
              <th className="py-1.5 font-medium">Records</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((d, i) => {
              const name = d.district ?? d.district_name ?? d.name ?? `District ${i + 1}`
              const risk = d.risk ?? d.risk_score ?? 0
              const conf = d.confidence ?? d.confidence_score ?? 0
              const count = d.records ?? d.work_count ?? d.works ?? 0
              return (
                <tr key={name + i} className="border-b border-navy/5 last:border-0 hover:bg-white/30">
                  <td className="py-2 text-navy/40">{i + 1}</td>
                  <td className="py-2 font-medium text-navy">{name}</td>
                  <td className="py-2 text-navy/70">{d.state ?? '—'}</td>
                  <td className={`py-2 font-semibold ${RISK_TEXT_COLOR(risk)}`}>
                    {risk.toFixed(2)}
                  </td>
                  <td className="py-2 text-navy/70">{conf.toFixed(2)}</td>
                  <td className="py-2 font-mono text-navy/70">{count.toLocaleString('en-IN')}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[10px] text-navy/40">
        Prioritized by Risk Score • Band: {riskBand(0.85)} threshold
      </p>
    </div>
  )
}

export function RecentHighRiskWorksTable() {
  const rows: WorkRecord[] = Array.isArray(recentHighRiskWorks)
    ? recentHighRiskWorks
    : (recentHighRiskWorks as any)()

  return (
    <div className="rounded-2xl border border-gold/30 bg-parchment-deep/60 p-4 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-serif text-lg text-navy">Recent High-Risk Works</h3>
        <a href="#works" className="text-xs font-medium text-navy/60 hover:text-navy">
          View All →
        </a>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-navy/10 text-navy/50">
              <th className="py-1.5 font-medium">Work ID</th>
              <th className="py-1.5 font-medium">Work Name</th>
              <th className="py-1.5 font-medium">District</th>
              <th className="py-1.5 font-medium">Sanctioned</th>
              <th className="py-1.5 font-medium">Risk</th>
              <th className="py-1.5 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((w) => (
              <tr key={w.work_id} className="border-b border-navy/5 last:border-0 hover:bg-white/30">
                <td className="py-2 font-mono text-[11px] font-semibold text-navy/70">{w.work_id}</td>
                <td className="py-2 font-medium text-navy">{w.work_name}</td>
                <td className="py-2 text-navy/70">{w.district}</td>
                <td className="py-2 font-mono text-navy/80">{formatINR(w.amount)}</td>
                <td className={`py-2 font-semibold ${RISK_TEXT_COLOR(w.risk)}`}>
                  {w.risk.toFixed(2)}
                </td>
                <td className="py-2">
                  <span
                    className={`rounded-full border px-2 py-0.5 text-[10px] font-medium shadow-xs ${
                      STATUS_PILL[w.status] || 'bg-gray-100 text-gray-700'
                    }`}
                  >
                    {w.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
