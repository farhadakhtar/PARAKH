import { Suspense, lazy, useEffect, useMemo, useState } from 'react'
import {
  featurePolygons,
  INDIA_BOUNDS as bounds,
  indiaGeo,
  getStateName,
  normalizeStateName,
} from '../data/indiaGeo'
import { fetchDistrictRisks, riskColor, RISK_BAND_COLORS } from '../data/mockData'
import type { DistrictRisk } from '../data/mockData'
import { CompassRose } from './svg/BrandIcons'

const Map3D = lazy(() => import('./Map3D'))

type ViewMode = '3D' | '2D'
type MapViewFilter =
  | 'Risk (R)'
  | 'Confidence (C)'
  | 'Decision'
  | 'Work Type'
  | 'Vendor Exposure'
  | 'Risk Signals'

interface Props {
  onSelectDistrict?: (d: DistrictRisk) => void
}

const VIEW_BY_OPTIONS: MapViewFilter[] = [
  'Risk (R)',
  'Confidence (C)',
  'Decision',
  'Work Type',
  'Vendor Exposure',
  'Risk Signals',
]

const KEY_2D_LABELS = [
  { name: 'Delhi', lon: 77.21, lat: 28.61, isPin: true },
  { name: 'Jammu & Kashmir', lon: 74.8, lat: 33.78 },
  { name: 'Ladakh', lon: 77.58, lat: 34.15 },
  { name: 'Himachal Pradesh', lon: 77.17, lat: 31.8 },
  { name: 'Punjab', lon: 75.34, lat: 31.15 },
  { name: 'Haryana', lon: 76.09, lat: 29.06 },
  { name: 'Rajasthan', lon: 73.8, lat: 26.5 },
  { name: 'Uttar Pradesh', lon: 80.94, lat: 26.85 },
  { name: 'Bihar', lon: 85.31, lat: 25.1 },
  { name: 'Madhya Pradesh', lon: 77.8, lat: 23.3 },
  { name: 'Gujarat', lon: 71.19, lat: 22.26 },
  { name: 'Maharashtra', lon: 75.71, lat: 19.75 },
  { name: 'Telangana', lon: 79.02, lat: 18.11 },
  { name: 'Chhattisgarh', lon: 81.87, lat: 21.28 },
  { name: 'Jharkhand', lon: 85.28, lat: 23.61 },
  { name: 'Odisha', lon: 84.46, lat: 20.95 },
  { name: 'West Bengal', lon: 87.85, lat: 23.4 },
  { name: 'Andhra Pradesh', lon: 79.74, lat: 15.91 },
  { name: 'Karnataka', lon: 75.71, lat: 15.32 },
  { name: 'Goa', lon: 74.12, lat: 15.3 },
  { name: 'Tamil Nadu', lon: 78.66, lat: 11.13 },
  { name: 'Kerala', lon: 76.27, lat: 10.85 },
  { name: 'Arunachal Pradesh', lon: 94.73, lat: 28.22 },
  { name: 'Meghalaya', lon: 91.37, lat: 25.47 },
  { name: 'Nagaland', lon: 94.56, lat: 26.16 },
  { name: 'Manipur', lon: 93.91, lat: 24.66 },
  { name: 'Mizoram', lon: 92.94, lat: 23.16 },
  { name: 'Tripura', lon: 91.99, lat: 23.94 },
  { name: 'Lakshadweep', lon: 72.78, lat: 10.57 },
  { name: 'Andaman & Nicobar Islands', lon: 92.66, lat: 11.74 },
]

export default function ChoroplethMap({ onSelectDistrict }: Props) {
  const [mode, setMode] = useState<ViewMode>('3D')
  const [viewBy, setViewBy] = useState<MapViewFilter>('Risk (R)')
  const [hovered, setHovered] = useState<string | null>(null)
  const [districtRisks, setDistrictRisks] = useState<DistrictRisk[]>([])

  useEffect(() => {
    const names = indiaGeo.features.map((f: any) => getStateName(f))
    fetchDistrictRisks(names).then((data) => setDistrictRisks(data))
  }, [])

  const riskByName = useMemo(() => {
    const m = new Map<string, DistrictRisk>()
    districtRisks.forEach((d) => {
      const raw = d.district_name ?? (d as any).name ?? ''
      const norm = normalizeStateName(raw)
      m.set(raw, d)
      m.set(norm, d)
      m.set(raw.replace(/\s*and\s*/gi, ' & '), d)
      m.set(raw.replace(/\s*&\s*/g, ' and '), d)
    })
    return m
  }, [districtRisks])

  const b = bounds
  const W = 720
  const H = 720
  const pad = 24

  const project = (lon: number, lat: number): [number, number] => {
    const x = pad + ((lon - b.minLon) / (b.maxLon - b.minLon)) * (W - pad * 2)
    const y = pad + (1 - (lat - b.minLat) / (b.maxLat - b.minLat)) * (H - pad * 2)
    return [x, y]
  }

  const pathFor = (feature: any): string => {
    const polys = featurePolygons(feature)
    if (!polys || !polys.length) return ''
    return polys
      .flatMap((poly) =>
        poly.map((ring: [number, number][]) => {
          const [start, ...rest] = ring.map(([lon, lat]) => project(lon, lat))
          if (!start) return ''
          return `M${start[0]},${start[1]} ` + rest.map(([x, y]) => `L${x},${y}`).join(' ') + ' Z'
        })
      )
      .join(' ')
  }

  const valueFor = (name: string): number => {
    const norm = normalizeStateName(name)
    const d = riskByName.get(name) || riskByName.get(norm)
    if (!d) return 0.5
    return viewBy === 'Confidence (C)'
      ? (d.confidence_score ?? (d as any).confidence ?? 0.7)
      : (d.risk_score ?? (d as any).risk ?? 0.5)
  }

  return (
    <div className="relative overflow-hidden rounded-xl border border-[#d8cbb0] bg-[#112330] shadow-sm">
      {/* 3D or 2D Map View Canvas */}
      {mode === '3D' ? (
        <Suspense
          fallback={
            <div className="flex h-[480px] items-center justify-center font-serif text-xs text-slate-300">
              Loading 3D geospatial terrain…
            </div>
          }
        >
          <Map3D
            geoData={indiaGeo as any}
            bounds={b}
            valueFor={valueFor}
            onSelectDistrict={(name) => {
              const d = riskByName.get(name)
              if (d) onSelectDistrict?.(d)
            }}
          />
        </Suspense>
      ) : (
        <div className="relative flex h-[480px] items-center justify-center p-2 bg-[#112330]">
          {/* Oceanic water names in 2D */}
          <div className="pointer-events-none absolute left-12 top-1/2 font-serif text-xs italic tracking-widest text-slate-400/40 select-none">
            Arabian Sea
          </div>
          <div className="pointer-events-none absolute right-14 top-1/2 font-serif text-xs italic tracking-widest text-slate-400/40 select-none">
            Bay of Bengal
          </div>
          <div className="pointer-events-none absolute bottom-10 left-1/2 -translate-x-1/2 font-serif text-xs italic tracking-widest text-slate-400/40 select-none">
            Indian Ocean
          </div>

          <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto h-full w-auto max-h-[460px]">
            {/* Base state polygons */}
            {indiaGeo.features.map((f: any, i: number) => {
              const name = getStateName(f) || `state-${i}`
              const v = valueFor(name)
              return (
                <path
                  key={name + i}
                  d={pathFor(f)}
                  fill={riskColor(v)}
                  stroke="#112330"
                  strokeWidth={0.8}
                  opacity={hovered && hovered !== name ? 0.45 : 0.95}
                  onMouseEnter={() => setHovered(name)}
                  onMouseLeave={() => setHovered(null)}
                  onClick={() => {
                    const d = riskByName.get(name)
                    if (d) onSelectDistrict?.(d)
                  }}
                  className="cursor-pointer transition-opacity"
                >
                  <title>{`${name}: Risk ${v.toFixed(2)}`}</title>
                </path>
              )
            })}

            {/* State Name Labels & Pins */}
            {KEY_2D_LABELS.map((item) => {
              const [px, py] = project(item.lon, item.lat)
              return (
                <g key={item.name} pointerEvents="none" className="select-none">
                  {item.isPin && (
                    <circle cx={px} cy={py} r={3} fill="#ef4444" stroke="#ffffff" strokeWidth={1} />
                  )}
                  <text
                    x={item.isPin ? px + 5 : px}
                    y={py + 3}
                    fontSize={item.isPin ? '9' : '7'}
                    fontWeight={item.isPin ? 'bold' : '600'}
                    fill={item.isPin ? '#ef4444' : '#ffffff'}
                    textAnchor={item.isPin ? 'start' : 'middle'}
                    className="drop-shadow-[0_1px_2px_rgba(0,0,0,0.95)]"
                  >
                    {item.name}
                  </text>
                </g>
              )
            })}
          </svg>
        </div>
      )}

      {/* FLOATING OVERLAY 1: VIEW MAP BY (Top-Left) */}
      <div className="absolute top-3 left-3 z-10 w-36 rounded-md border border-white/20 bg-[#fdfbf7]/90 p-2 shadow-md backdrop-blur-xs">
        <h4 className="mb-1.5 text-[9px] font-bold tracking-wider text-slate-800 uppercase">
          VIEW MAP BY
        </h4>
        <div className="space-y-0.5">
          {VIEW_BY_OPTIONS.map((opt) => {
            const isSelected = viewBy === opt
            return (
              <button
                key={opt}
                onClick={() => setViewBy(opt)}
                className={`flex w-full items-center gap-1.5 rounded px-1 py-0.5 text-left text-[10px] font-medium transition-colors ${
                  isSelected
                    ? 'bg-[#edd8a4]/70 text-[#0b1a2d] font-bold'
                    : 'text-slate-700 hover:bg-black/5'
                }`}
              >
                <span
                  className={`flex h-2.5 w-2.5 items-center justify-center rounded-full border ${
                    isSelected
                      ? 'border-[#c9a227] bg-[#c9a227]'
                      : 'border-slate-400 bg-white'
                  }`}
                >
                  {isSelected && <span className="h-1 w-1 rounded-full bg-white" />}
                </span>
                <span>{opt}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* FLOATING OVERLAY 2: RISK LEVEL (Middle-Left, directly below VIEW MAP BY) */}
      <div className="absolute top-44 left-3 z-10 w-36 rounded-md border border-white/20 bg-[#fdfbf7]/90 p-2 shadow-md backdrop-blur-xs">
        <h4 className="mb-1.5 text-[9px] font-bold tracking-wider text-slate-800 uppercase">
          RISK LEVEL
        </h4>
        <div className="space-y-1 text-[10px]">
          {Object.entries(RISK_BAND_COLORS).map(([band, hex]) => (
            <div key={band} className="flex items-center gap-1.5 text-slate-700">
              <span
                className="h-2 w-2 rounded-full shrink-0"
                style={{ backgroundColor: hex }}
              />
              <span className="font-medium">{band}</span>
            </div>
          ))}
        </div>
      </div>

      {/* FLOATING OVERLAY 3: Compass Rose (Bottom-Left) */}
      <div className="absolute bottom-3 left-3 z-10">
        <CompassRose size={36} />
      </div>

      {/* FLOATING OVERLAY 4: Interaction Prompt (Bottom-Center) */}
      <div className="pointer-events-none absolute bottom-2.5 left-1/2 -translate-x-1/2 z-10 select-none">
        <div className="rounded-full bg-black/60 px-2.5 py-0.5 text-[9px] font-medium text-slate-200 backdrop-blur-xs shadow-xs">
          👆 Hover over a district • Scroll to zoom • Click to explore
        </div>
      </div>

      {/* FLOATING OVERLAY 5: 3D / 2D Switcher Pill (Bottom-Right) */}
      <div className="absolute bottom-3 right-3 z-10 flex overflow-hidden rounded border border-slate-300 bg-white shadow-md">
        <button
          onClick={() => setMode('3D')}
          className={`px-2.5 py-0.5 text-[11px] font-bold transition-all ${
            mode === '3D'
              ? 'bg-[#0b1a2d] text-white'
              : 'bg-white text-slate-700 hover:bg-slate-100'
          }`}
        >
          3D
        </button>
        <button
          onClick={() => setMode('2D')}
          className={`px-2.5 py-0.5 text-[11px] font-bold transition-all ${
            mode === '2D'
              ? 'bg-[#0b1a2d] text-white'
              : 'bg-white text-slate-700 hover:bg-slate-100'
          }`}
        >
          2D
        </button>
      </div>
    </div>
  )
}
