import { useState } from 'react'
import Sidebar from './components/layout/Sidebar'
import TopBar from './components/layout/TopBar'
import ChoroplethMap from './components/ChoroplethMap'
import DecisionMatrix from './components/DecisionMatrix'
import DispositionKpi from './components/DispositionKpi'
import { TopRiskDistrictsTable, RecentHighRiskWorksTable } from './components/SurveillanceTables'
import {
  Database,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  Calendar,
  MapPin,
  ChevronDown,
  Shield,
  FileCheck,
  Sparkles,
} from 'lucide-react'
import {
  LionCapitalEmblem,
  RashtrapatiBhavanEngraving,
  TricolorWave,
} from './components/svg/BrandIcons'
import type { DistrictRisk } from './data/mockData'

const KPIS = [
  {
    icon: Database,
    value: '20,000',
    label: 'Total Works Analysed',
    trend: '↑ 184 this week',
    trendColor: 'text-[#16a34a]',
    iconColor: 'text-[#1d4ed8]',
    iconBg: 'bg-[#dbeafe]',
  },
  {
    icon: ShieldCheck,
    value: '78.6%',
    label: 'Avg. Evidentiary Confidence',
    trend: '↑ 6.3% vs last week',
    trendColor: 'text-[#16a34a]',
    iconColor: 'text-[#1d4ed8]',
    iconBg: 'bg-[#dbeafe]',
  },
  {
    icon: AlertTriangle,
    value: '382',
    label: 'High Risk Records',
    trend: '↑ 23 vs last week',
    trendColor: 'text-[#dc2626]',
    iconColor: 'text-[#dc2626]',
    iconBg: 'bg-[#fee2e2]',
  },
  {
    icon: CheckCircle2,
    value: '18,476',
    label: 'Records Cleared',
    trend: '↑ 112 vs last week',
    trendColor: 'text-[#16a34a]',
    iconColor: 'text-[#16a34a]',
    iconBg: 'bg-[#dcfce7]',
  },
]

const SYSTEM_STATUS_ITEMS = [
  { name: 'Data Pipeline', status: 'Operational' },
  { name: 'Risk Models', status: 'Healthy' },
  { name: 'Confidence Engine', status: 'Healthy' },
  { name: 'Evidence Store', status: 'Synced' },
  { name: 'API Services', status: 'Online' },
]

export default function App() {
  const [selected, setSelected] = useState<DistrictRisk | null>(null)

  const selectedName = selected?.district_name ?? (selected as any)?.name
  const selectedRisk = selected?.risk_score ?? (selected as any)?.risk ?? 0
  const selectedConf = selected?.confidence_score ?? (selected as any)?.confidence ?? 0
  const selectedCount = selected?.work_count ?? (selected as any)?.workCount ?? 0

  return (
    <div className="flex h-screen overflow-hidden bg-[#f4ece1] font-sans text-slate-800 antialiased">
      {/* 1. Left Sidebar */}
      <Sidebar />

      {/* 2. Main Content Area */}
      <div className="flex flex-1 flex-col overflow-y-auto bg-[#f6f0e2]">
        {/* Top Bar Header */}
        <TopBar />

        {/* Dashboard Body */}
        <main className="flex-1 space-y-3.5 p-4 lg:p-5">
          {/* Subheader: Greeting, PM Quote, Date & Region Pickers */}
          <div className="flex flex-col justify-between gap-3 lg:flex-row lg:items-center">
            {/* Left Greeting */}
            <div>
              <h2 className="font-serif text-xl font-bold text-[#0b1a2d]">
                Namaste, Auditor 👋
              </h2>
              <p className="text-[11px] text-slate-600 font-normal">
                Turning public data into public trust.
              </p>
            </div>

            {/* Center Quote */}
            <div className="text-center font-serif text-xs text-[#0b1a2d]">
              <span className="italic text-xs font-semibold">
                “Transparent governance builds a stronger nation.”
              </span>
              <div className="text-[9px] text-slate-600 not-italic mt-0.5">
                — Hon'ble Prime Minister
              </div>
            </div>

            {/* Right: Date Range & Location Dropdowns */}
            <div className="flex items-center gap-2">
              <button className="flex items-center gap-1.5 rounded-lg border border-[#d8cbb0] bg-[#fbf9f4] px-2.5 py-1 text-[11px] font-medium text-slate-700 shadow-xs hover:bg-white">
                <Calendar size={12} className="text-slate-500" />
                <span>12 May 2024 – 18 May 2024</span>
                <ChevronDown size={12} className="text-slate-400" />
              </button>

              <button className="flex items-center gap-1.5 rounded-lg border border-[#d8cbb0] bg-[#fbf9f4] px-2.5 py-1 text-[11px] font-medium text-slate-700 shadow-xs hover:bg-white">
                <MapPin size={12} className="text-slate-500" />
                <span>India (All States)</span>
                <ChevronDown size={12} className="text-slate-400" />
              </button>
            </div>
          </div>

          {/* Row 1: 5 KPI Cards (4 Metrics + 1 Viksit Bharat Banner) */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {KPIS.map((k) => (
              <div
                key={k.label}
                className="flex items-center gap-3 rounded-xl border border-[#d8cbb0] bg-[#fbf9f4] p-3 shadow-xs transition hover:shadow-sm"
              >
                <div
                  className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-slate-200/60 ${k.iconBg}`}
                >
                  <k.icon className={k.iconColor} size={18} />
                </div>
                <div className="flex flex-col leading-tight">
                  <span className="font-serif text-lg font-bold text-[#0b1a2d]">
                    {k.value}
                  </span>
                  <span className="text-[10px] font-medium text-slate-500">
                    {k.label}
                  </span>
                  <span className={`text-[9.5px] font-bold mt-0.5 ${k.trendColor}`}>
                    {k.trend}
                  </span>
                </div>
              </div>
            ))}

            {/* 5th Banner Card: Viksit Bharat */}
            <div className="relative flex items-center justify-between overflow-hidden rounded-xl border border-[#d8cbb0] bg-gradient-to-r from-[#f7eed9] via-[#faefe0] to-[#fbf9f4] p-3 shadow-xs">
              <div className="shrink-0 opacity-80 pl-0.5">
                <LionCapitalEmblem size={38} color="#bfa15f" />
              </div>

              <div className="flex flex-col text-right leading-tight pr-0.5">
                <span className="font-serif text-[12px] font-bold tracking-wide text-[#5c4015] italic">
                  Viksit Bharat
                </span>
                <span className="font-serif text-[10px] font-semibold text-[#78551c]">
                  Sashakt Niyat
                </span>
                <span className="font-serif text-[12px] font-bold tracking-tight text-[#4a3411] italic">
                  Samriddh Bharat
                </span>
                <div className="mt-1 w-20 self-end">
                  <div className="h-0.5 w-full bg-gradient-to-r from-[#ff9933] via-[#ffffff] to-[#138808] rounded-full" />
                </div>
              </div>
            </div>
          </div>

          {/* Row 2: Map (2 Cols) + Surveillance Tables (1 Col) */}
          <div className="grid grid-cols-1 gap-3.5 lg:grid-cols-3">
            {/* Map Column */}
            <div className="lg:col-span-2 flex flex-col gap-2">
              <ChoroplethMap onSelectDistrict={setSelected} />

              {selected && (
                <div className="flex items-center justify-between rounded-lg border border-[#c9a227] bg-[#fbf9f4] px-3.5 py-1.5 text-xs shadow-sm">
                  <div className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-[#c9a227]" />
                    <span className="font-serif font-bold text-[#0b1a2d]">
                      {selectedName}
                    </span>
                    <span className="text-slate-600 text-[11px]">
                      — Risk: <strong className="text-red-600">{selectedRisk.toFixed(2)}</strong> ·
                      Confidence:{' '}
                      <strong className="text-blue-600">{selectedConf.toFixed(2)}</strong> · Works:{' '}
                      <strong>{selectedCount.toLocaleString('en-IN')}</strong>
                    </span>
                  </div>
                  <button
                    onClick={() => setSelected(null)}
                    className="text-slate-400 hover:text-slate-700 text-xs"
                  >
                    ✕
                  </button>
                </div>
              )}
            </div>

            {/* Tables Column */}
            <div className="space-y-3">
              <TopRiskDistrictsTable />
              <RecentHighRiskWorksTable />
            </div>
          </div>

          {/* Row 3: Bottom Analytics Row (4 Cards) */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {/* 1. Decision Distribution */}
            <DispositionKpi />

            {/* 2. Risk vs Confidence Matrix */}
            <DecisionMatrix />

            {/* 3. System Status */}
            <div className="flex flex-col justify-between rounded-xl border border-[#d8cbb0] bg-[#fbf9f4] p-3.5 shadow-xs">
              <div>
                <h3 className="mb-2 font-serif text-xs font-bold text-[#0b1a2d]">
                  System Status
                </h3>
                <div className="space-y-1 text-xs">
                  {SYSTEM_STATUS_ITEMS.map((item) => (
                    <div
                      key={item.name}
                      className="flex items-center justify-between text-slate-700"
                    >
                      <span className="flex items-center gap-1.5 font-medium text-[11px]">
                        <span className="h-1.5 w-1.5 rounded-full bg-[#16a34a] shadow-[0_0_3px_#22c55e]" />
                        {item.name}
                      </span>
                      <span className="font-semibold text-[10.5px] text-[#15803d]">
                        {item.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-2.5 border-t border-[#e2d5bd] pt-1.5 text-[9.5px] text-slate-500">
                Last updated: 18 May 2024, 09:30 AM
              </div>
            </div>

            {/* 4. Monument Quote Card */}
            <div className="relative flex flex-col justify-between overflow-hidden rounded-xl border border-[#d8cbb0] bg-[#fbf9f4] p-3.5 shadow-xs">
              <div className="pointer-events-none absolute -bottom-3 -right-3 w-40 opacity-25">
                <RashtrapatiBhavanEngraving opacity={0.65} />
              </div>

              <div className="relative z-10">
                <blockquote className="font-serif text-base font-bold leading-snug text-[#2c1d06] italic">
                  “Public money
                  <br />
                  builds public trust.”
                </blockquote>
              </div>

              <div className="relative z-10 mt-5 flex items-center justify-between">
                <span className="font-serif text-[11px] font-semibold tracking-wider text-[#8b6528]">
                  Viksit Bharat
                </span>
                <div className="w-14">
                  <div className="h-0.5 w-full bg-gradient-to-r from-[#ff9933] via-[#ffffff] to-[#138808] rounded-full" />
                </div>
              </div>
            </div>
          </div>
        </main>

        {/* Institutional Bottom Footer */}
        <footer className="mt-2 border-t border-[#d8cbb0] bg-[#f7eedc] px-5 py-2.5 shrink-0">
          <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-slate-700">
            {/* 1. Transparency */}
            <div className="flex items-center gap-1.5">
              <Shield size={14} className="text-[#8b6528]" />
              <div className="flex flex-col leading-tight">
                <span className="font-bold text-[11px] text-[#0b1a2d]">Transparency</span>
                <span className="text-[9px] text-slate-500">Open by default</span>
              </div>
            </div>

            {/* 2. Accountability */}
            <div className="flex items-center gap-1.5">
              <FileCheck size={14} className="text-[#8b6528]" />
              <div className="flex flex-col leading-tight">
                <span className="font-bold text-[11px] text-[#0b1a2d]">Accountability</span>
                <span className="text-[9px] text-slate-500">Data for people</span>
              </div>
            </div>

            {/* 3. Better Governance */}
            <div className="flex items-center gap-1.5">
              <Sparkles size={14} className="text-[#8b6528]" />
              <div className="flex flex-col leading-tight">
                <span className="font-bold text-[11px] text-[#0b1a2d]">Better Governance</span>
                <span className="text-[9px] text-slate-500">Insights for impact</span>
              </div>
            </div>

            {/* 4. Ministry of Finance */}
            <div className="flex items-center gap-2">
              <LionCapitalEmblem size={24} color="#78551c" />
              <div className="flex flex-col leading-tight">
                <span className="font-bold text-[11px] text-[#0b1a2d]">Government of India</span>
                <span className="text-[9px] text-slate-500">
                  Ministry of Finance
                </span>
                <span className="text-[8px] text-slate-400">
                  (Department of Expenditure)
                </span>
              </div>
            </div>

            {/* 5. Satyamev Jayate */}
            <div className="flex flex-col items-end leading-tight">
              <span className="font-serif text-[11px] font-bold text-[#0b1a2d]">
                सत्यमेव जयते
              </span>
              <span className="text-[9px] font-medium text-slate-500">
                Satyamev Jayate
              </span>
              <div className="w-14 mt-0.5">
                <TricolorWave className="h-1.5 w-14" />
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  )
}
