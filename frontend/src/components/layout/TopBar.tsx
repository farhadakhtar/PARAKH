import { useState } from 'react'
import {
  Search,
  Bell,
  LayoutDashboard,
  Compass,
  BarChart2,
  Layers,
  FileSpreadsheet,
} from 'lucide-react'
import { TricolorWave } from '../svg/BrandIcons'

export default function TopBar() {
  const [activeTab, setActiveTab] = useState('Overview')

  const TABS = [
    { id: 'Overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'Explore', label: 'Explore', icon: Compass },
    { id: 'Analytics', label: 'Analytics', icon: BarChart2 },
    { id: 'Records', label: 'Records', icon: Layers },
    { id: 'Reports', label: 'Reports', icon: FileSpreadsheet },
  ]

  return (
    <header className="shrink-0 border-b border-[#d8cbb0]/80 bg-[#f9f5ec] px-5 py-2.5 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
      <div className="flex items-center justify-between gap-3">
        {/* Left: Brand / Title */}
        <div className="flex flex-col shrink-0">
          <h1 className="font-serif text-xl font-extrabold tracking-[0.2em] text-[#0b1a2d]">
            PARAKH
          </h1>
          <p className="text-[9px] font-medium leading-tight text-slate-500">
            Public Asset & Risk Assessment for
            <br />
            Knowledge-driven Highways / Works
          </p>
        </div>

        {/* Center: Navigation Pill Tabs */}
        <nav className="flex items-center gap-1 rounded-full border border-[#d8cbb0]/80 bg-[#ede4cf]/50 p-1 shadow-inner shrink-0">
          {TABS.map(({ id, label, icon: Icon }) => {
            const isActive = activeTab === id
            return (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-medium transition-all ${
                  isActive
                    ? 'bg-[#ebd8a6] text-[#0b1a2d] font-bold shadow-xs'
                    : 'text-slate-600 hover:text-[#0b1a2d] hover:bg-white/40'
                }`}
              >
                <Icon size={12} className={isActive ? 'text-[#0b1a2d]' : 'text-slate-500'} />
                <span>{label}</span>
              </button>
            )
          })}
        </nav>

        {/* Right Section: Search, Notification, User, Satyamev Jayate */}
        <div className="flex items-center gap-3 shrink-0">
          {/* Search Box */}
          <div className="relative flex items-center">
            <Search size={13} className="absolute left-2.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search district, work ID, vendor..."
              className="h-7 w-48 rounded-full border border-[#d8cbb0] bg-[#fbf9f4] pl-8 pr-2.5 text-[11px] text-slate-800 placeholder:text-slate-400 shadow-inner focus:border-[#c9a227] focus:bg-white focus:outline-none"
            />
          </div>

          {/* Bell Notification with red count badge */}
          <button className="relative rounded-full p-1 text-slate-600 transition hover:bg-black/5 hover:text-slate-900">
            <Bell size={17} />
            <span className="absolute -top-1 -right-1 flex h-3.5 w-3.5 items-center justify-center rounded-full bg-[#e53e3e] text-[8px] font-bold text-white shadow-xs">
              3
            </span>
          </button>

          {/* User Profile Chip */}
          <div className="flex items-center gap-2 pl-1">
            <div className="flex h-7 w-7 items-center justify-center rounded-full bg-[#0b1a2d] text-[11px] font-bold text-white shadow-xs">
              A
            </div>
            <div className="flex flex-col leading-tight">
              <span className="text-[11px] font-bold text-[#0b1a2d]">Auditor</span>
              <span className="text-[9px] text-slate-500 font-medium">Delhi Circle</span>
            </div>
          </div>

          {/* Satyamev Jayate + Tricolor Ribbon */}
          <div className="flex flex-col items-end border-l border-[#d8cbb0]/80 pl-3">
            <span className="font-serif text-[10px] font-bold tracking-wider text-[#0b1a2d]">
              सत्यमेव जयते
            </span>
            <span className="text-[8px] font-medium text-slate-500 -mt-0.5">
              Satyamev Jayate
            </span>
            <div className="w-18 -mt-0.5">
              <TricolorWave className="h-2.5 w-18" />
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
