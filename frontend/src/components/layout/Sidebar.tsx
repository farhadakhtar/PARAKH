import {
  Home,
  MapPin,
  FolderSearch,
  BarChart3,
  FileSearch,
  Database,
  Users,
  Settings,
} from 'lucide-react'

const NAV_ITEMS = [
  { icon: Home, label: 'Home', active: true },
  { icon: MapPin, label: 'Spatial Intelligence' },
  { icon: FolderSearch, label: 'Works Explorer' },
  { icon: BarChart3, label: 'Risk Analysis' },
  { icon: FileSearch, label: 'Evidence & Reports' },
]

const NAV_ITEMS_SECONDARY = [
  { icon: Database, label: 'Data & Tools' },
  { icon: Users, label: 'User Management' },
  { icon: Settings, label: 'System Settings' },
]

export default function Sidebar() {
  return (
    <aside className="flex h-full w-64 shrink-0 flex-col bg-navy px-4 py-6 text-parchment">
      <div className="mb-8 flex flex-col items-center text-center">
        {/* Ashoka chakra inspired emblem */}
        <div className="mb-2 flex h-14 w-14 items-center justify-center rounded-full border-2 border-gold bg-navy-light text-xl text-gold shadow-inner">
          ⚖️
        </div>
        <span className="font-serif text-sm font-bold tracking-wider text-gold">PARAKH</span>
        <span className="text-[9px] tracking-widest text-gold/70">GOVERNMENT OF INDIA</span>
      </div>

      <nav className="flex-1 space-y-1">
        {NAV_ITEMS.map(({ icon: Icon, label, active }) => (
          <button
            key={label}
            className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
              active
                ? 'border-l-2 border-gold bg-gold/10 font-semibold text-gold'
                : 'text-parchment/70 hover:bg-white/5 hover:text-parchment'
            }`}
          >
            <Icon size={16} />
            {label}
          </button>
        ))}
        <div className="my-3 border-t border-white/10" />
        {NAV_ITEMS_SECONDARY.map(({ icon: Icon, label }) => (
          <button
            key={label}
            className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-parchment/70 transition hover:bg-white/5 hover:text-parchment"
          >
            <Icon size={16} />
            {label}
          </button>
        ))}
      </nav>

      <div className="mt-6 border-t border-white/10 pt-4 text-center">
        <div className="mx-auto mb-2 text-lg opacity-80">🏛️</div>
        <p className="text-[10px] leading-relaxed text-gold/80">
          जनहित में पारदर्शिता
          <br />
          For a Stronger, More Accountable India
        </p>
      </div>
    </aside>
  )
}
