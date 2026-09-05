import { Search, Bell, ChevronDown } from 'lucide-react'

export default function TopBar() {
  return (
    <header className="flex items-center justify-between border-b border-gold/20 bg-parchment px-6 py-3">
      <div>
        <div className="flex items-center gap-2">
          <h1 className="font-serif text-2xl font-bold text-navy">PARAKH</h1>
          <span className="rounded bg-gold/20 px-2 py-0.5 text-[10px] font-bold text-gold">v2.0</span>
        </div>
        <p className="text-[11px] text-navy/60">
          Pattern Analysis & Relationship Analytics for Knowledge-based Heuristics
        </p>
      </div>

      <nav className="hidden gap-6 text-sm font-medium text-navy/60 md:flex">
        {['Overview', 'Explore', 'Analytics', 'Records', 'Reports'].map((item, i) => (
          <span
            key={item}
            className={i === 0 ? 'border-b-2 border-gold pb-1 font-semibold text-navy' : 'cursor-pointer hover:text-navy'}
          >
            {item}
          </span>
        ))}
      </nav>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 rounded-full border border-navy/15 bg-white/60 px-3 py-1.5 text-sm text-navy/70 shadow-sm">
          <Search size={14} className="text-navy/40" />
          <input
            placeholder="Search district, work ID, vendor..."
            className="w-48 bg-transparent outline-none placeholder:text-navy/40"
          />
        </div>
        <div className="relative cursor-pointer">
          <Bell size={18} className="text-navy/70 hover:text-navy" />
          <span className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-[9px] font-bold text-white shadow">
            3
          </span>
        </div>
        <div className="flex items-center gap-2 border-l border-navy/10 pl-3 text-sm text-navy">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-navy text-xs font-semibold text-parchment shadow-sm">
            A
          </div>
          <div className="leading-tight">
            <p className="font-medium">Aditya (Auditor)</p>
            <p className="text-[10px] text-navy/50">Delhi Circle</p>
          </div>
          <ChevronDown size={14} className="text-navy/40" />
        </div>
      </div>
    </header>
  )
}
