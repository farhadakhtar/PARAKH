/**
 * PARAKH frontend data layer.
 *
 * No working backend server exists on this branch yet (Stage 7 ui_app.py /
 * FastAPI service are not implemented). Every fetch below returns locally
 * generated mock data with a TODO marking the real endpoint it should call
 * once the Stage 1-7 pipeline exposes an HTTP API.
 */

export const API_BASE = import.meta.env.VITE_PARAKH_API_URL ?? 'http://localhost:8000/api/v1'

export interface DistrictRisk {
  district_name: string
  risk_score: number // R, 0..1
  confidence_score: number // C, 0..1
  work_count: number
}

export interface WorkRecord {
  work_id: string
  work_name: string
  district: string
  amount: number // INR
  risk: number
  status: 'Investigate' | 'Remediate' | 'Monitor'
}

export interface DecisionDistribution {
  decision: 'Investigate' | 'Remediate' | 'Monitor' | 'Clear'
  count: number
}

export interface RiskConfidencePoint {
  risk: number
  confidence: number
  decision: 'Investigate' | 'Remediate' | 'Monitor' | 'Clear'
}

/* ---------- deterministic seeded PRNG so the demo is stable ---------- */
function hashSeed(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}
function mulberry32(seed: number) {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Per-district (currently per-state) risk data for the map.
 * TODO: replace with real call: GET ${API_BASE}/map/districts?level=district
 * returning DistrictRisk[] for every district in the corpus.
 */
export async function fetchDistrictRisks(stateNames: string[]): Promise<DistrictRisk[]> {
  void API_BASE
  return stateNames.map((name) => {
    const rand = mulberry32(hashSeed(name))
    // Bimodal-ish spread so the choropleth shows hot and cold regions.
    const risk = Math.min(0.98, Math.max(0.02, rand() * 0.55 + rand() * rand() * 0.45))
    const confidence = Math.min(0.99, Math.max(0.35, 0.55 + rand() * 0.45))
    const work_count = Math.round(200 + rand() * 4800)
    return { district_name: name, risk_score: risk, confidence_score: confidence, work_count }
  })
}

/** TODO: replace with GET ${API_BASE}/districts/top-risk?limit=5 */
export const topRiskDistricts: Array<{
  rank: number
  district: string
  state: string
  risk: number
  confidence: number
  records: number
}> = [
  { rank: 1, district: 'Sonbhadra', state: 'UP', risk: 0.92, confidence: 0.81, records: 214 },
  { rank: 2, district: 'Gadchiroli', state: 'MH', risk: 0.88, confidence: 0.76, records: 186 },
  { rank: 3, district: 'Latehar', state: 'JH', risk: 0.84, confidence: 0.69, records: 143 },
  { rank: 4, district: 'Koraput', state: 'OD', risk: 0.81, confidence: 0.73, records: 120 },
  { rank: 5, district: 'Barmer', state: 'RJ', risk: 0.78, confidence: 0.66, records: 118 },
]

/** TODO: replace with GET ${API_BASE}/works/high-risk?limit=5 */
export const recentHighRiskWorks: WorkRecord[] = [
  {
    work_id: 'WORK-18392',
    work_name: 'Drain construction',
    district: 'Delhi',
    amount: 8000000,
    risk: 0.91,
    status: 'Investigate',
  },
  {
    work_id: 'WORK-17621',
    work_name: 'Road repair',
    district: 'Sonbhadra',
    amount: 4500000,
    risk: 0.87,
    status: 'Investigate',
  },
  {
    work_id: 'WORK-16211',
    work_name: 'School repair',
    district: 'Koraput',
    amount: 1200000,
    risk: 0.68,
    status: 'Remediate',
  },
  {
    work_id: 'WORK-15890',
    work_name: 'Bridge development',
    district: 'Gadchiroli',
    amount: 2500000,
    risk: 0.66,
    status: 'Remediate',
  },
  {
    work_id: 'WORK-14932',
    work_name: 'Water supply',
    district: 'Latehar',
    amount: 1800000,
    risk: 0.62,
    status: 'Monitor',
  },
]

/**
 * Decision counts feed the donut chart and the KPI strip.
 * TODO: replace with GET ${API_BASE}/decisions/distribution
 */
export const decisionDistribution: DecisionDistribution[] = [
  { decision: 'Investigate', count: 382 },
  { decision: 'Remediate', count: 1142 },
  { decision: 'Monitor', count: 6586 },
  { decision: 'Clear', count: 12890 },
]
export const TOTAL_WORKS = decisionDistribution.reduce((s, d) => s + d.count, 0) // 20,000

/**
 * Scatter points for the Risk-vs-Confidence quadrant chart.
 * High-density realistic distribution matching the reference UI scatter plot.
 */
export function riskConfidencePoints(seed = 108): RiskConfidencePoint[] {
  const rand = mulberry32(seed)
  const pts: RiskConfidencePoint[] = []

  // Generate dense realistic cluster matching the reference chart:
  // 1. Monitor (Top-Left: R in 0.08-0.44, C in 0.52-0.96, green/yellow)
  for (let i = 0; i < 90; i++) {
    const r = 0.08 + rand() * 0.36
    const c = 0.52 + rand() * 0.44
    pts.push({ risk: r, confidence: c, decision: 'Monitor' })
  }

  // 2. Investigate (Top-Right: R in 0.52-0.96, C in 0.52-0.96, red)
  for (let i = 0; i < 80; i++) {
    const r = 0.52 + rand() * 0.44
    const c = 0.52 + rand() * 0.44
    pts.push({ risk: r, confidence: c, decision: 'Investigate' })
  }

  // 3. Clear (Bottom-Left: dense band around R: 0.05-0.45, C: 0.05-0.48, green)
  for (let i = 0; i < 120; i++) {
    const r = 0.05 + rand() * 0.40
    const c = 0.05 + rand() * 0.44
    pts.push({ risk: r, confidence: c, decision: 'Clear' })
  }

  // 4. Remediate (Bottom-Right: R in 0.52-0.94, C in 0.08-0.48, orange)
  for (let i = 0; i < 70; i++) {
    const r = 0.52 + rand() * 0.42
    const c = 0.08 + rand() * 0.40
    pts.push({ risk: r, confidence: c, decision: 'Remediate' })
  }

  return pts
}

/* ---------- shared risk colour scale (used by 2D + 3D maps) ---------- */
export type RiskBand = 'Very High' | 'High' | 'Medium' | 'Low' | 'Very Low'

export function riskBand(r: number): RiskBand {
  if (r >= 0.8) return 'Very High'
  if (r >= 0.6) return 'High'
  if (r >= 0.4) return 'Medium'
  if (r >= 0.2) return 'Low'
  return 'Very Low'
}

export const RISK_BAND_COLORS: Record<RiskBand, string> = {
  'Very High': '#c0392b',
  High: '#e67e22',
  Medium: '#f1c40f',
  Low: '#84c442',
  'Very Low': '#1e8449',
}

/** Smooth red -> orange -> yellow -> light-green -> dark-green ramp, t in 0..1 */
export function riskColor(t: number): string {
  const stops: Array<[number, [number, number, number]]> = [
    [0, [30, 132, 73]],
    [0.25, [132, 196, 66]],
    [0.5, [241, 196, 15]],
    [0.75, [230, 126, 34]],
    [1, [192, 57, 43]],
  ]
  const clamped = Math.min(1, Math.max(0, t))
  for (let i = 1; i < stops.length; i++) {
    if (clamped <= stops[i][0]) {
      const [t0, c0] = stops[i - 1]
      const [t1, c1] = stops[i]
      const f = (clamped - t0) / (t1 - t0)
      const mix = c0.map((v, j) => Math.round(v + (c1[j] - v) * f)) as [number, number, number]
      return `rgb(${mix[0]},${mix[1]},${mix[2]})`
    }
  }
  return '#c0392b'
}

export const formatINR = (n: number): string => `₹${n.toLocaleString('en-IN')}`

/** Formats amount in Lakhs (e.g. 8000000 -> ₹80L) matching the reference UI */
export const formatAmountLakhs = (n: number): string => {
  const lakhs = n / 100000
  if (Number.isInteger(lakhs)) return `₹${lakhs}L`
  return `₹${lakhs.toFixed(1)}L`
}
