import { Suspense, lazy, useEffect, useMemo, useState } from 'react'
import { featurePolygons, INDIA_BOUNDS as bounds, indiaGeo, getStateName, normalizeStateName } from '../data/indiaGeo'
import { fetchDistrictRisks, riskColor, RISK_BAND_COLORS } from '../data/mockData'
import type { DistrictRisk } from '../data/mockData'

const Map3D = lazy(() => import('./Map3D'))

type ViewMode = '2D' | '3D'
type ColorBy = 'risk' | 'confidence'

interface Props {
  onSelectDistrict?: (d: DistrictRisk) => void
}

export default function ChoroplethMap({ onSelectDistrict }: Props) {
  const [mode, setMode] = useState<ViewMode>('3D')
  const [colorBy, setColorBy] = useState<ColorBy>('risk')
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
  const W = 760
  const H = 760
  const pad = 25

  // Equirectangular projection into SVG viewBox space
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
    return colorBy === 'risk'
      ? (d.risk_score ?? (d as any).risk ?? 0.5)
      : (d.confidence_score ?? (d as any).confidence ?? 0.7)
  }

  return (
    <div className="relative rounded-2xl border border-gold/30 bg-parchment-deep/60 p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs">
          {(['risk', 'confidence'] as ColorBy[]).map((c) => (
            <button
              key={c}
              onClick={() => setColorBy(c)}
              className={`rounded-full border px-3 py-1 capitalize transition ${
                colorBy === c
                  ? 'border-gold bg-gold/20 font-semibold text-navy'
                  : 'border-navy/20 text-navy/60 hover:border-gold/50'
              }`}
            >
              {c === 'risk' ? 'Risk Score (R)' : 'Confidence (C)'}
            </button>
          ))}
        </div>
        <div className="flex overflow-hidden rounded-full border border-navy/20">
          {(['3D', '2D'] as ViewMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-4 py-1 text-xs font-semibold transition ${
                mode === m ? 'bg-navy text-parchment' : 'bg-transparent text-navy/60 hover:text-navy'
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {mode === '2D' ? (
        <div className="flex items-center justify-center p-2">
          <svg viewBox={`0 0 ${W} ${H}`} className="mx-auto w-full max-w-[560px]">
            {indiaGeo.features.map((f: any, i: number) => {
              const name = getStateName(f) || `state-${i}`
              const v = valueFor(name)
              return (
                <path
                  key={name + i}
                  d={pathFor(f)}
                  fill={riskColor(v)}
                  stroke="#f5efe0"
                  strokeWidth={1}
                  opacity={hovered && hovered !== name ? 0.45 : 1}
                  onMouseEnter={() => setHovered(name)}
                  onMouseLeave={() => setHovered(null)}
                  onClick={() => {
                    const d = riskByName.get(name)
                    if (d) onSelectDistrict?.(d)
                  }}
                  className="cursor-pointer transition-opacity"
                >
                  <title>{`${name}: ${colorBy === 'risk' ? 'Risk' : 'Confidence'} ${v.toFixed(2)}`}</title>
                </path>
              )
            })}
          </svg>
        </div>
      ) : (
        <Suspense
          fallback={
            <div className="flex h-[560px] items-center justify-center text-navy/50">
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
      )}

      <p className="mt-3 text-center text-xs text-navy/50">
        Hover over a state/district • Scroll to zoom • Click to inspect evidence
      </p>

      <div className="mt-3 flex flex-wrap justify-center gap-3 text-[11px]">
        {Object.entries(RISK_BAND_COLORS).map(([band, hex]) => (
          <span key={band} className="flex items-center gap-1.5 text-navy/70">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: hex as string }}
            />
            {band}
          </span>
        ))}
      </div>
    </div>
  )
}
