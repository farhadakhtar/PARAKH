# PARAKH: Frontend Architecture & Implementation Status Report
**Current Branch:** `frontend-clean`  
**Generated:** September 2026

---

## 1. Executive Summary

The frontend for **PARAKH** has been rebuilt and scaffolded on the `frontend-clean` branch, migrating from an earlier prototype to a modern **React 19 + Tailwind CSS v4 + Three.js / React Three Fiber + Recharts** architecture.

The foundation is designed around an **Indian governance & forensic audit aesthetic** (parchment, deep navy, gold, and tricolor accents) with native support for:
1. **Interactive Geospatial Risk Choropleth** (2D SVG & 3D Three.js extruded map using GeoJSON).
2. **$2 \times 2$ Confidence-Gated Decision Visualizer** (Risk vs. Confidence scatter plot).
3. **Forensic Audit KPI & Disposition Strip** (Donut breakdown of `Investigate`, `Remediate`, `Monitor`, `Clear`).
4. **Actionable High-Risk Works & Top Districts Surveillance Tables**.

---

## 2. Technology Stack & Dependencies

Defined in [`frontend/package.json`](file:///c:/Users/chakr/PARAKH/frontend/package.json) and [`frontend/vite.config.ts`](file:///c:/Users/chakr/PARAKH/frontend/vite.config.ts):

| Dependency Category | Packages | Version | Role in PARAKH |
| :--- | :--- | :--- | :--- |
| **Core Framework** | `react`, `react-dom` | `^19.2.8` | Component state, UI rendering |
| **Build Tooling** | `vite`, `@vitejs/plugin-react` | `^8.2.2` | Lightning-fast HMR and bundle compilation |
| **Styling & Theme** | `tailwindcss`, `@tailwindcss/vite` | `^4.3.3` | Tailwind v4 with CSS `@theme` tokens |
| **3D Map Visualization** | `three`, `@react-three/fiber`, `@react-three/drei` | `^0.185.1` / `^9.7.0` | 3D extruded choropleth map of Indian states |
| **Charts & Analytics** | `recharts` | `^3.10.1` | Donut disposition charts, $2 \times 2$ scatter plot |
| **Iconography** | `lucide-react` | `^1.41.0` | Clean, accessible dashboard icons |
| **Linting & Quality** | `oxlint` | `^1.79.0` | High-performance Rust-based JavaScript/TypeScript linter |
| **Language** | `typescript` | `~6.0.2` | Full end-to-end type safety |

---

## 3. Directory Structure & Files Created

```
frontend/
├── package.json                   # React 19, Tailwind v4, Three.js, Recharts, Vite 8
├── vite.config.ts                 # React + Tailwind plugins configuration
├── tsconfig.json                  # Root TypeScript configuration
├── tsconfig.app.json              # App-specific TS settings
├── tsconfig.node.json             # Node-specific TS settings
├── .oxlintrc.json                 # Oxlint configuration
├── .gitignore                     # Local ignore rules
├── index.html                     # HTML5 shell
├── public/
│   ├── favicon.svg                # Application icon
│   └── icons.svg                  # SVG sprite sheet
└── src/
    ├── main.tsx                   # React root mount
    ├── index.css                  # Bespoke theme tokens, fonts, and scrollbars
    ├── App.tsx                    # Current view (Vite starter template, ready for dashboard wire-up)
    ├── App.css                    # Starter CSS rules
    ├── assets/
    │   ├── hero.png               # Brand asset
    │   ├── react.svg, vite.svg    # Starter vectors
    │   └── india-states.geojson   # 1.27 MB simplified GADM level-1 GeoJSON boundary data
    ├── data/
    │   ├── indiaGeo.ts            # Polygon ring extraction, bounds, and projection math
    │   └── mockData.ts            # Data contracts, seeded PRNG, mock data, and color scales
    └── components/
        └── svg/                   # SVG subcomponents container
```

---

## 4. Completed Subsystems & Modules

### 4.1. Design System & Theming ([`frontend/src/index.css`](file:///c:/Users/chakr/PARAKH/frontend/src/index.css))
A curated governance theme replacing generic dark/light defaults with an Indian institutional palette:
* **Typography:**
  * Serif Header Font: `'Playfair Display', Georgia, serif` (authoritative, legal-brief feel)
  * Body Sans Font: `'Inter', system-ui, sans-serif` (clean, high legibility for metrics)
* **Tailwind v4 Theme Tokens:**
  * `--color-parchment: #f5efe0` (warm archival background)
  * `--color-parchment-deep: #ede4cf` (card and container fills)
  * `--color-navy: #0d1b2a` (deep midnight text & structural accents)
  * `--color-navy-light: #16304a` (subtle framing)
  * `--color-gold: #c9a227` / `--color-gold-light: #e6cd7a` (emblem and highlight accents)
  * `--color-saffron: #ff9933` / `--color-india-green: #138808` (national tricolor palette)
* **Custom UI Utilities:**
  * `.scroll-slim`: Custom scrollbars with translucent gold thumbs for data-dense tables.
  * `.tricolor-underline`: Linear gradient (Saffron $\to$ White $\to$ India Green) for title bars.

---

### 4.2. Geographic Data & Projection Engine ([`frontend/src/data/indiaGeo.ts`](file:///c:/Users/chakr/PARAKH/frontend/src/data/indiaGeo.ts))
* **GeoJSON Ingestion:** Ingests [`india-states.geojson`](file:///c:/Users/chakr/PARAKH/frontend/src/assets/india-states.geojson) (1.27 MB) with polygon and multipolygon definitions.
* **Geometry Processing:**
  * `featurePolygons(f: GeoFeature)`: Extracts outer rings and hole polygons for every Indian state.
  * `bounds`: Dynamically computes bounding coordinates (`minLon`, `maxLon`, `minLat`, `maxLat`) to center both 2D SVG maps and 3D WebGL scenes.
  * **Projection Ready:** Formatted for both SVG path generation and Three.js `ExtrudeGeometry` for height-elevated risk choropleths.

---

### 4.3. Data Layer & Risk Calibration ([`frontend/src/data/mockData.ts`](file:///c:/Users/chakr/PARAKH/frontend/src/data/mockData.ts))
* **Type Contracts:**
  * `DistrictRisk`: Name, Risk ($R \in [0, 1]$), Confidence ($C \in [0, 1]$), work volume.
  * `WorkRecord`: `work_id`, `work_name`, `district`, `amount`, `risk`, `status` (`Investigate` | `Remediate` | `Monitor`).
  * `DecisionDistribution`: `decision` (`Investigate` | `Remediate` | `Monitor` | `Clear`), `count`.
  * `RiskConfidencePoint`: Risk, Confidence, Decision quadrant mapping.
* **Deterministic PRNG Engine:**
  * `mulberry32` + `hashSeed`: Generates consistent, repeatable mock distributions without hydration jitter during live demos.
* **Analytical Datasets:**
  * `fetchDistrictRisks`: Regional risk/confidence mapping across all states.
  * `topRiskDistricts`: Ranked surveillance table (Gaya, Jaunpur, Nagaur, Nashik, Paschim Medinipur).
  * `recentHighRiskWorks`: Sample high-value works with anomaly statuses.
  * `decisionDistribution`: Aggregated count (382 `Investigate`, 1,142 `Remediate`, 6,586 `Monitor`, 12,890 `Clear` out of 20,000 total works).
  * `riskConfidencePoints`: 200 synthetic points clustered across the 4 quadrants for scatter plotting.
* **Color Ramps & Formatting:**
  * `riskBand(r)` & `RISK_BAND_COLORS`: 5 tiers (`Very High` `#c0392b`, `High` `#e67e22`, `Medium` `#f1c40f`, `Low` `#a9df8a`, `Very Low` `#1e8449`).
  * `riskColor(t)`: Smooth RGB linear interpolation between color stops for continuous choropleths.
  * `formatINR(n)`: Formats Indian Rupee currency (e.g., `₹18,50,000`).

---

## 5. Component Assembly & Build Verification

All 5 core dashboard components, supporting layout shells, and the integrated `App.tsx` have been built, validated against the data layer contracts, and compiled via `npm run build`.

| Component / Feature | Path | Implementation Status | Features Verified |
| :--- | :--- | :---: | :--- |
| **Dependencies & Tooling** | `package.json`, `vite.config.ts` | **Completed** | Vite 8, React 19, Tailwind v4, Three.js, Recharts, Oxlint |
| **Theme & Design Tokens** | `src/index.css` | **Completed** | Playfair Display, Inter, Parchment, Deep Navy, Gold, Tricolor |
| **India GeoJSON & Math** | `src/data/indiaGeo.ts` | **Completed** | GADM GeoJSON, decimation, bounds & projections |
| **Data Contracts & Mocks** | `src/data/mockData.ts` | **Completed** | Seeded PRNG, types, color ramps, formatters |
| **Sidebar Navigation** | `src/components/layout/Sidebar.tsx` | **Completed** | Ashoka chakra branding, primary & secondary navigation |
| **Top Header Bar** | `src/components/layout/TopBar.tsx` | **Completed** | Auditor profile, notifications, search bar, circle branding |
| **3D Geospatial Scene** | `src/components/Map3D.tsx` | **Completed** | Lazy-loaded Three.js extruded height-risk terrain, decimation, OrbitControls |
| **Choropleth Map (2D/3D)** | `src/components/ChoroplethMap.tsx` | **Completed** | 2D SVG equirectangular + 3D toggle, Risk/Confidence color switches, tooltips |
| **$2 \times 2$ Decision Matrix** | `src/components/DecisionMatrix.tsx` | **Completed** | Recharts scatter plot with 4-quadrant reference lines, tooltips |
| **Disposition KPI Strip** | `src/components/DispositionKpi.tsx` | **Completed** | Recharts donut chart with central work count and legend percentages |
| **Surveillance Tables** | `src/components/SurveillanceTables.tsx` | **Completed** | Top 5 risk districts & recent high-risk works tables |
| **Dashboard Assembly** | `src/App.tsx` | **Completed** | Integrated KPI strip, interactive map, tables, matrix, system status |

---

## 6. Build & Bundle Metrics

Ran production build check (`tsc -b && vite build`):
* **Status:** `0 errors, exit code 0`
* **Modules Transformed:** `2,965 modules`
* **Compilation Time:** `2.50s`
* **Code Splitting:**
  * `Map3D` dynamically imported as an on-demand chunk (`Map3D-*.js`, 238 kB gzip) so 2D map view requires zero Three.js bundle overhead.
  * Core bundle: `index-*.js` (674 kB gzip) and `index-*.css` (5.6 kB gzip).

