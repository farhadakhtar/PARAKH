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
  { rank: 1, district: 'Gaya', state: 'Bihar', risk: 0.94, confidence: 0.91, records: 612 },
  { rank: 2, district: 'Jaunpur', state: 'Uttar Pradesh', risk: 0.91, confidence: 0.88, records: 845 },
  { rank: 3, district: 'Nagaur', state: 'Rajasthan', risk: 0.87, confidence: 0.92, records: 530 },
  { rank: 4, district: 'Nashik', state: 'Maharashtra', risk: 0.84, confidence: 0.86, records: 477 },
  { rank: 5, district: 'Paschim Medinipur', state: 'West Bengal', risk: 0.82, confidence: 0.9, records: 661 },
]

/** TODO: replace with GET ${API_BASE}/works/high-risk?limit=5 */
export const recentHighRiskWorks: WorkRecord[] = [
  {
    work_id: 'WRK-26-88134',
    work_name: 'Village Road CC Pavement',
    district: 'Gaya, BR',
    amount: 1850000,
    risk: 0.95,
    status: 'Investigate',
  },
  {
    work_id: 'WRK-26-87902',
    work_name: 'Drinking Water Pipeline',
    district: 'Jaunpur, UP',
    amount: 2430000,
    risk: 0.91,
    status: 'Investigate',
  },
  {
    work_id: 'WRK-26-87571',
    work_name: 'Anganwadi Centre Renovation',
    district: 'Nagaur, RJ',
    amount: 960000,
    risk: 0.86,
    status: 'Remediate',
  },
  {
    work_id: 'WRK-26-87119',
    work_name: 'Farm Pond Excavation',
    district: 'Nashik, MH',
    amount: 720000,
    risk: 0.83,
    status: 'Monitor',
  },
  {
    work_id: 'WRK-26-86985',
    work_name: 'Gram Panchayat Bhawan',
    district: 'Medinipur, WB',
    amount: 3110000,
    risk: 0.81,
    status: 'Investigate',
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
 * TODO: replace with GET ${API_BASE}/records/scatter?sample=200
 */
export function riskConfidencePoints(seed = 42, n = 200): RiskConfidencePoint[] {
  const rand = mulberry32(seed)
  const pts: RiskConfidencePoint[] = []
  const clusters: Array<[number, number, RiskConfidencePoint['decision'], number]> = [
    [0.8, 0.78, 'Investigate', 0.14], // high C, high R
    [0.2, 0.8, 'Monitor', 0.14], //    high C, low R
    [0.78, 0.2, 'Remediate', 0.15], // low C, high R
    [0.22, 0.22, 'Clear', 0.16], //    low C, low R
  ]
  for (let i = 0; i < n; i++) {
    const [cx, cy, decision, spread] = clusters[i % clusters.length]
    const jitter = () => (rand() + rand() - 1) * spread
    pts.push({
      risk: Math.min(0.99, Math.max(0.02, cx + jitter())),
      confidence: Math.min(0.99, Math.max(0.02, cy + jitter())),
      decision,
    })
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
  Low: '#a9df8a',
  'Very Low': '#1e8449',
}

/** Smooth red -> orange -> yellow -> light-green -> dark-green ramp, t in 0..1 */
export function riskColor(t: number): string {
  const stops: Array<[number, [number, number, number]]> = [
    [0, [30, 132, 73]],
    [0.25, [169, 223, 138]],
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
