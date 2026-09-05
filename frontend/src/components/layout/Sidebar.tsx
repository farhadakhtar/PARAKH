import type { LucideIcon } from 'lucide-react'
import {
  Home,
  Hexagon,
  FileText,
  TrendingUp,
  FileCheck2,
  Briefcase,
  Users,
  Settings,
  ChevronRight,
} from 'lucide-react'
import {
  LionCapitalEmblem,
  SansadBhavanIllustration,
  DigitalIndiaLogo,
} from '../svg/BrandIcons'

interface NavItem {
  icon: LucideIcon
  label: string
  active?: boolean
}

const PRIMARY_NAV: NavItem[] = [
  { icon: Home, label: 'Home', active: true },
  { icon: Hexagon, label: 'Spatial Intelligence' },
  { icon: FileText, label: 'Works Explorer' },
  { icon: TrendingUp, label: 'Risk Analysis' },
  { icon: FileCheck2, label: 'Evidence & Reports' },
]

const SECONDARY_NAV: NavItem[] = [
  { icon: Briefcase, label: 'Data & Tools' },
  { icon: Users, label: 'User Management' },
  { icon: Settings, label: 'System Settings' },
]

export default function Sidebar() {
  return (
    <aside className="relative flex h-screen w-64 shrink-0 flex-col justify-between border-r border-[#1a2d42] bg-[#0b1a2d] px-4 py-5 text-slate-200">
      {/* Top Header & Emblem */}
      <div>
        {/* Purple dot badge on top-left */}
        <div className="flex items-center gap-2 mb-2 px-1">
          <span className="h-2.5 w-2.5 rounded-full bg-[#6366f1] shadow-[0_0_8px_#6366f1]" />
        </div>

        <div className="mb-6 flex flex-col items-center text-center">
          <LionCapitalEmblem size={52} color="#d4af37" className="drop-shadow-md" />
          <span className="mt-1.5 font-serif text-sm font-semibold tracking-wider text-white">
            भारत सरकार
          </span>
          <span className="text-[9px] font-medium tracking-[0.18em] text-slate-300">
            GOVERNMENT OF INDIA
          </span>
        </div>

        {/* Primary Navigation */}
        <nav className="space-y-1.5">
          {PRIMARY_NAV.map(({ icon: Icon, label, active }) => (
            <button
              key={label}
              className={`group relative flex w-full items-center justify-between rounded-lg px-3.5 py-2.5 text-xs font-medium transition-all ${
                active
                  ? 'bg-gradient-to-r from-[#6b4c19] via-[#7d591e] to-[#8d6522] text-white shadow-md'
                  : 'text-slate-300 hover:bg-white/5 hover:text-white'
              }`}
            >
              <div className="flex items-center gap-3">
                <Icon
                  size={17}
                  className={active ? 'text-[#f5d061]' : 'text-slate-400 group-hover:text-slate-200'}
                />
                <span className={active ? 'font-semibold text-white' : ''}>{label}</span>
              </div>
              {active && (
                <ChevronRight size={14} className="text-[#f5d061]" />
              )}
            </button>
          ))}

          {/* Nav Section Divider */}
          <div className="py-2">
            <div className="h-px w-full bg-slate-700/60" />
          </div>

          {/* Secondary Navigation */}
          {SECONDARY_NAV.map(({ icon: Icon, label }) => (
            <button
              key={label}
              className="group flex w-full items-center gap-3 rounded-lg px-3.5 py-2.5 text-xs font-medium text-slate-300 transition-all hover:bg-white/5 hover:text-white"
            >
              <Icon size={17} className="text-slate-400 group-hover:text-slate-200" />
              <span>{label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Bottom Footer Section: Sansad Bhavan + Motto + Digital India */}
      <div className="mt-4 flex flex-col items-center pt-3 text-center">
        {/* Sansad Bhavan line drawing */}
        <div className="w-full max-w-[190px] px-2 opacity-85">
          <SansadBhavanIllustration color="#8fa3bf" />
        </div>

        <div className="mt-2.5 space-y-0.5">
          <p className="font-serif text-[11px] font-medium tracking-wide text-[#e6cd7a]">
            जनहित में पारदर्शिता
          </p>
          <p className="text-[10px] leading-tight text-slate-300/90 font-serif italic">
            For a Stronger,
            <br />
            More Accountable India
          </p>
        </div>

        {/* Digital India Brand */}
        <div className="mt-3.5 flex items-center justify-center border-t border-slate-700/50 pt-3">
          <DigitalIndiaLogo />
        </div>
      </div>
    </aside>
  )
}
