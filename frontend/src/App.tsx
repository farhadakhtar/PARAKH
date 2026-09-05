import { useState } from 'react'
import Sidebar from './components/layout/Sidebar'
import TopBar from './components/layout/TopBar'
import ChoroplethMap from './components/ChoroplethMap'
import DecisionMatrix from './components/DecisionMatrix'
import DispositionKpi from './components/DispositionKpi'
import { TopRiskDistrictsTable, RecentHighRiskWorksTable } from './components/SurveillanceTables'
import { Database, ShieldCheck, AlertTriangle, CheckCircle2 } from 'lucide-react'
import type { DistrictRisk } from './data/mockData'

const KPIS = [
  {
    icon: Database,
    value: '20,000',
    label: 'Total Works Analysed',
    trend: '↑ 184 this week',
    color: 'text-blue-600',
    trendColor: 'text-green-600',
  },
  {
    icon: ShieldCheck,
    value: '78.6%',
    label: 'Avg. Evidentiary Confidence',
    trend: '↑ 6.3% vs last week',
    color: 'text-emerald-600',
    trendColor: 'text-green-600',
  },
  {
    icon: AlertTriangle,
    value: '382',
    label: 'High Risk Records',
    trend: '↑ 23 vs last week',
    color: 'text-red-600',
    trendColor: 'text-red-600',
  },
  {
    icon: CheckCircle2,
    value: '18,476',
    label: 'Records Cleared',
    trend: '↑ 112 vs last week',
    color: 'text-green-600',
    trendColor: 'text-green-600',
  },
]

export default function App() {
  const [selected, setSelected] = useState<DistrictRisk | null>(null)

  const selectedName = selected?.district_name ?? (selected as any)?.name
  const selectedRisk = selected?.risk_score ?? (selected as any)?.risk ?? 0
  const selectedConf = selected?.confidence_score ?? (selected as any)?.confidence ?? 0
  const selectedCount = selected?.work_count ?? (selected as any)?.workCount ?? 0

  return (
    <div className="flex h-screen overflow-hidden bg-parchment font-sans text-navy">
      <Sidebar />

      <div className="flex flex-1 flex-col overflow-y-auto">
        <TopBar />

        <main className="flex-1 space-y-6 p-6">
          {/* Greeting banner */}
          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-start">
            <div>
              <h2 className="font-serif text-2xl font-bold text-navy">Namaste, Auditor 👋</h2>
              <p className="text-sm text-navy/60">
                Turning self-certified public fund data into actionable evidentiary leads.
              </p>
            </div>
            <div className="max-w-xs rounded-xl border border-gold/30 bg-parchment-deep/60 px-4 py-2 text-right text-xs italic text-navy/70 shadow-xs">
              "Transparent governance builds a stronger, more accountable nation."
              <div className="mt-1 text-[10px] font-medium not-italic text-gold">
                — Government of India • Audit Circle
              </div>
            </div>
          </div>

          {/* KPI strip */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {KPIS.map((k) => (
              <div
                key={k.label}
                className="rounded-2xl border border-gold/30 bg-parchment-deep/60 p-4 shadow-sm transition hover:border-gold/60"
              >
                <k.icon className={`mb-2 ${k.color}`} size={22} />
                <div className="font-serif text-2xl font-bold text-navy">{k.value}</div>
                <div className="text-xs text-navy/60">{k.label}</div>
                <div className={`mt-1 text-[11px] font-medium ${k.trendColor}`}>{k.trend}</div>
              </div>
            ))}
          </div>

          {/* Map + Side Tables */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <ChoroplethMap onSelectDistrict={setSelected} />
              {selected && (
                <div className="mt-3 flex items-center justify-between rounded-xl border border-gold/40 bg-white/70 px-4 py-2.5 text-sm shadow-sm">
                  <div className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-gold" />
                    <span className="font-serif font-bold text-navy">{selectedName}</span>
                    <span className="text-xs text-navy/60">
                      — Risk: <strong className="text-red-600">{selectedRisk.toFixed(2)}</strong> · Confidence:{' '}
                      <strong className="text-emerald-600">{selectedConf.toFixed(2)}</strong> · Total Works:{' '}
                      <strong>{selectedCount.toLocaleString('en-IN')}</strong>
                    </span>
                  </div>
                  <button
                    onClick={() => setSelected(null)}
                    className="text-xs text-navy/40 hover:text-navy"
                  >
                    ✕ Dismiss
                  </button>
                </div>
              )}
            </div>
            <div className="space-y-6">
              <TopRiskDistrictsTable />
              <RecentHighRiskWorksTable />
            </div>
          </div>

          {/* Bottom Analytics Row */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <DispositionKpi />
            <DecisionMatrix />
            <div className="rounded-2xl border border-gold/30 bg-parchment-deep/60 p-4 shadow-sm">
              <h3 className="mb-2 font-serif text-lg text-navy">System Status</h3>
              <ul className="space-y-2 text-xs">
                {[
                  ['Data Ingestion & Schema', 'Operational (Stage 1)'],
                  ['Evidentiary Confidence Kernel', 'Active (Stage 2)'],
                  ['Peer Cost Outliers & MAD', 'Operational (Stage 3)'],
                  ['Entity Linkage & Resolution', 'Active (Stage 4)'],
                  ['Vendor HHI & Burst Engine', 'Operational (Stage 5)'],
                  ['2×2 Policy Decision Engine', 'Active (Stage 6)'],
                  ['Artifact Invariance Test', 'Passed 99.0% (Stage 7)'],
                ].map(([label, status]) => (
                  <li key={label} className="flex items-center justify-between border-b border-navy/5 pb-1 last:border-0">
                    <span className="flex items-center gap-2 text-navy/70">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 shadow-xs" />
                      {label}
                    </span>
                    <span className="font-medium text-emerald-700">{status}</span>
                  </li>
                ))}
              </ul>
              <p className="mt-3 text-[10px] text-navy/40">
                Local Time: {new Date().toLocaleTimeString('en-IN')} • All modules synchronized
              </p>
            </div>
          </div>
        </main>

        <footer className="flex flex-wrap items-center justify-between border-t border-gold/20 bg-parchment px-6 py-3 text-[11px] text-navy/60">
          <span>Transparency — Open by Default</span>
          <span>Accountability — Data for People</span>
          <span>Better Governance — Insights for Impact</span>
          <span>Government of India, Ministry of Finance (Department of Expenditure)</span>
          <span className="font-serif font-semibold text-gold">सत्यमेव जयते</span>
        </footer>
      </div>
    </div>
  )
}
