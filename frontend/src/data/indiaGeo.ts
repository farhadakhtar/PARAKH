/**
 * Shared India states geographic data (Survey of India constitutional boundary compliant)
 * plus projection math shared by the 2D SVG map and 3D extruded map.
 */
import rawGeoJson from '../assets/india-states.geojson?raw'

export interface GeoFeature {
  type: 'Feature'
  properties: {
    NAME_1?: string
    st_nm?: string
    name?: string
    [k: string]: unknown
  }
  geometry: {
    type: 'Polygon' | 'MultiPolygon'
    coordinates: number[][][] | number[][][][]
  }
}

interface GeoCollection {
  type: 'FeatureCollection'
  features: GeoFeature[]
}

export const indiaGeo = JSON.parse(rawGeoJson) as GeoCollection

export type LngLatRing = Array<[number, number]>

/**
 * Normalization mapping between official Census/Survey of India state names,
 * common abbreviations, and mock data strings.
 */
export const STATE_NAME_ALIASES: Record<string, string> = {
  // Jammu & Kashmir and Ladakh
  'Jammu & Kashmir': 'Jammu and Kashmir',
  'Jammu & Kashmir (UT)': 'Jammu and Kashmir',
  'J&K': 'Jammu and Kashmir',
  'J & K': 'Jammu and Kashmir',
  'UT of Jammu and Kashmir': 'Jammu and Kashmir',
  'UT of Ladakh': 'Ladakh',
  'Ladakh (UT)': 'Ladakh',

  // Island territories and enclave mergers
  'Andaman and Nicobar': 'Andaman and Nicobar Islands',
  'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
  'Andaman & Nicobar': 'Andaman and Nicobar Islands',
  'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli',
  'Daman & Diu': 'Daman and Diu',
  'Dadra and Nagar Haveli and Daman and Diu': 'Dadra and Nagar Haveli',
  'DNHDD': 'Dadra and Nagar Haveli',

  // Capital territory & renamed states
  'NCT of Delhi': 'Delhi',
  'National Capital Territory of Delhi': 'Delhi',
  'Orissa': 'Odisha',
  'Uttaranchal': 'Uttarakhand',
  'Pondicherry': 'Puducherry',
}

/**
 * Normalizes any variation of an Indian State/UT name to its canonical name.
 */
export function normalizeStateName(rawName: string): string {
  if (!rawName) return ''
  const trimmed = rawName.trim()
  if (STATE_NAME_ALIASES[trimmed]) {
    return STATE_NAME_ALIASES[trimmed]
  }
  // Try substituting '&' with 'and'
  const withAnd = trimmed.replace(/\s*&\s*/g, ' and ')
  if (STATE_NAME_ALIASES[withAnd]) {
    return STATE_NAME_ALIASES[withAnd]
  }
  return trimmed
}

/**
 * Extracts and normalizes the state/UT name from a GeoFeature.
 */
export function getStateName(f: GeoFeature): string {
  const raw = f.properties?.st_nm ?? f.properties?.NAME_1 ?? f.properties?.name ?? ''
  return normalizeStateName(raw)
}

/** Flatten a feature into a list of polygon ring-lists (holes included). */
export function featurePolygons(f: GeoFeature): LngLatRing[][] {
  if (f.geometry.type === 'Polygon') {
    return [
      (f.geometry.coordinates as number[][][]).map(
        (ring) => ring.map((p) => [p[0], p[1]]) as LngLatRing
      ),
    ]
  }
  return (f.geometry.coordinates as number[][][][]).map((poly) =>
    poly.map((ring) => ring.map((p) => [p[0], p[1]]) as LngLatRing)
  )
}

/* Bounds of the full collection (covering 68.2°E to 97.4°E, 6.7°N to 37.1°N) */
const computedBounds = (() => {
  let minLon = Infinity,
    maxLon = -Infinity,
    minLat = Infinity,
    maxLat = -Infinity
  for (const f of indiaGeo.features) {
    for (const poly of featurePolygons(f)) {
      for (const ring of poly) {
        for (const [lon, lat] of ring) {
          if (lon < minLon) minLon = lon
          if (lon > maxLon) maxLon = lon
          if (lat < minLat) minLat = lat
          if (lat > maxLat) maxLat = lat
        }
      }
    }
  }
  return {
    minLon,
    maxLon,
    minLat,
    maxLat,
    centerLon: (minLon + maxLon) / 2,
    centerLat: (minLat + maxLat) / 2,
  }
})()

export const bounds = computedBounds
export const INDIA_BOUNDS = computedBounds

/** World-units map sized to roughly FIT_SIZE wide, centered on origin. */
export const MAP_SCALE = 110 / (bounds.maxLon - bounds.minLon)
const cosMid = Math.cos(((bounds.minLat + bounds.maxLat) / 2) * (Math.PI / 180))

/**
 * lon/lat -> planar map coords. x east, y north.
 * For the 3D scene: use x for X, then extrude-shape rotation puts y onto -Z,
 * i.e. north (+lat) appears as -Z (top of screen with the default camera).
 */
export function project(lon: number, lat: number): [number, number] {
  return [
    (lon - bounds.centerLon) * MAP_SCALE * cosMid,
    (bounds.centerLat - lat) * MAP_SCALE,
  ]
}

/** Map-size helpers for the 2D SVG viewBox. */
export const MAP_SIZE = {
  width: (bounds.maxLon - bounds.minLon) * MAP_SCALE * cosMid,
  height: (bounds.maxLat - bounds.minLat) * MAP_SCALE,
}

/** Mathematical centroid (area-weighted polygon centroid) of a GeoFeature. */
export function getFeatureCentroidLonLat(f: GeoFeature): [number, number] {
  const polys = featurePolygons(f)
  if (!polys || !polys.length) return [bounds.centerLon, bounds.centerLat]
  let maxArea = -1
  let bestCx = bounds.centerLon
  let bestCy = bounds.centerLat

  for (const poly of polys) {
    const ring = poly[0]
    if (!ring || ring.length < 3) continue
    let area = 0
    let cx = 0
    let cy = 0
    for (let i = 0; i < ring.length - 1; i++) {
      const x0 = ring[i][0]
      const y0 = ring[i][1]
      const x1 = ring[i + 1][0]
      const y1 = ring[i + 1][1]
      const a = x0 * y1 - x1 * y0
      area += a
      cx += (x0 + x1) * a
      cy += (y0 + y1) * a
    }
    area *= 0.5
    const absArea = Math.abs(area)
    if (absArea > maxArea) {
      maxArea = absArea
      if (absArea > 1e-6) {
        bestCx = cx / (6 * area)
        bestCy = cy / (6 * area)
      } else {
        let sx = 0
        let sy = 0
        for (const p of ring) {
          sx += p[0]
          sy += p[1]
        }
        bestCx = sx / ring.length
        bestCy = sy / ring.length
      }
    }
  }
  return [bestCx, bestCy]
}


/** Major states to label on the map */
export const STATE_LABELS: Array<{
  feature: string
  text: [number, number] // anchor as lon/lat (hand-tuned)
}> = [
  { feature: 'Ladakh', text: [77.5, 34.5] },
  { feature: 'Jammu and Kashmir', text: [74.8, 33.7] },
  { feature: 'Rajasthan', text: [73.2, 26.9] },
  { feature: 'Uttar Pradesh', text: [80.9, 26.8] },
  { feature: 'Madhya Pradesh', text: [78.3, 23.5] },
  { feature: 'Bihar', text: [85.6, 25.8] },
  { feature: 'Maharashtra', text: [76.2, 19.3] },
  { feature: 'Karnataka', text: [76.4, 15.2] },
  { feature: 'Gujarat', text: [71.4, 22.7] },
  { feature: 'West Bengal', text: [87.8, 23.6] },
  { feature: 'Tamil Nadu', text: [78.4, 10.9] },
  { feature: 'Andhra Pradesh', text: [79.6, 15.6] },
  { feature: 'Telangana', text: [78.9, 17.7] },
  { feature: 'Odisha', text: [84.6, 20.5] },
]
