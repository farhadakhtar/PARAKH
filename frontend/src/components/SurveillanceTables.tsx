import { topRiskDistricts, recentHighRiskWorks, formatAmountLakhs } from '../data/mockData'
import type { WorkRecord } from '../data/mockData'

const STATUS_BADGES: Record<string, string> = {
  Investigate: 'bg-[#fee2e2] text-[#b91c1c] border-[#fca5a5]',
  Remediate: 'bg-[#ffedd5] text-[#c2410c] border-[#fdba74]',
  Monitor: 'bg-[#fef9c3] text-[#854d0e] border-[#fde047]',
}

export function TopRiskDistrictsTable() {
  return (
    <div className="rounded-xl border border-[#d8cbb0] bg-[#fbf9f4] p-3 shadow-xs">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-serif text-xs font-bold text-[#0b1a2d]">Top Risk Districts</h3>
        <a
          href="#districts"
          className="flex items-center gap-0.5 text-[11px] font-semibold text-[#1e3a8a] transition hover:text-[#0b1a2d]"
        >
          <span>View All</span>
          <span>→</span>
        </a>
      </div>
      <div className="w-full">
        <table className="w-full text-left text-[11px]">
          <thead>
            <tr className="border-b border-[#e2d5bd] text-[10px] font-medium text-slate-500">
              <th className="py-1 px-1 font-medium w-5">#</th>
              <th className="py-1 px-1.5 font-medium">District</th>
              <th className="py-1 px-1 font-medium">State</th>
              <th className="py-1 px-1 font-medium text-center">Risk (R)</th>
              <th className="py-1 px-1 font-medium text-center">Confidence (C)</th>
              <th className="py-1 px-1 font-medium text-right">Records</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#ebdcc4]/40">
            {topRiskDistricts.map((d) => (
              <tr key={d.district} className="hover:bg-black/[0.015]">
                <td className="py-1.5 px-1 text-slate-400 font-mono text-[10px]">{d.rank}</td>
                <td className="py-1.5 px-1.5 font-medium text-[#0b1a2d] whitespace-nowrap">{d.district}</td>
                <td className="py-1.5 px-1 text-slate-600 font-medium whitespace-nowrap">{d.state}</td>
                <td className="py-1.5 px-1 font-bold text-[#dc2626] text-center">{d.risk.toFixed(2)}</td>
                <td className="py-1.5 px-1 font-semibold text-[#2563eb] text-center">{d.confidence.toFixed(2)}</td>
                <td className="py-1.5 px-1 text-right font-medium text-slate-700">{d.records}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function RecentHighRiskWorksTable() {
  return (
    <div className="rounded-xl border border-[#d8cbb0] bg-[#fbf9f4] p-3 shadow-xs">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="font-serif text-xs font-bold text-[#0b1a2d]">Recent High-Risk Works</h3>
        <a
          href="#works"
          className="flex items-center gap-0.5 text-[11px] font-semibold text-[#1e3a8a] transition hover:text-[#0b1a2d]"
        >
          <span>View All</span>
          <span>→</span>
        </a>
      </div>
      <div className="w-full">
        <table className="w-full text-left text-[11px]">
          <thead>
            <tr className="border-b border-[#e2d5bd] text-[10px] font-medium text-slate-500">
              <th className="py-1 px-1 font-medium">Work ID</th>
              <th className="py-1 px-1.5 font-medium">Work Name</th>
              <th className="py-1 px-1 font-medium">District</th>
              <th className="py-1 px-1 font-medium">Amount</th>
              <th className="py-1 px-1 font-medium text-center">Risk</th>
              <th className="py-1 px-1 font-medium text-right">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#ebdcc4]/40">
            {recentHighRiskWorks.map((w: WorkRecord) => (
              <tr key={w.work_id} className="hover:bg-black/[0.015]">
                <td className="py-1.5 px-1 font-mono text-[10px] font-medium text-slate-700 whitespace-nowrap">
                  {w.work_id}
                </td>
                <td className="py-1.5 px-1.5 font-medium text-[#0b1a2d] whitespace-nowrap">{w.work_name}</td>
                <td className="py-1.5 px-1 text-slate-600 whitespace-nowrap">{w.district}</td>
                <td className="py-1.5 px-1 font-medium text-slate-800 whitespace-nowrap">
                  {formatAmountLakhs(w.amount)}
                </td>
                <td className="py-1.5 px-1 font-bold text-[#dc2626] text-center">{w.risk.toFixed(2)}</td>
                <td className="py-1.5 px-1 text-right whitespace-nowrap">
                  <span
                    className={`inline-block rounded border px-1.5 py-0.5 text-[9px] font-semibold ${
                      STATUS_BADGES[w.status] || 'bg-slate-100 text-slate-700'
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
