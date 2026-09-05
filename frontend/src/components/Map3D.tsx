import { useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import * as THREE from 'three'
import {
  featurePolygons,
  getStateName,
  normalizeStateName,
  getFeatureCentroidLonLat,
} from '../data/indiaGeo'
import { riskColor } from '../data/mockData'

interface Bounds {
  minLon: number
  maxLon: number
  minLat: number
  maxLat: number
}

interface Props {
  geoData: { features: any[] }
  bounds: Bounds
  valueFor: (name: string) => number
  onSelectDistrict?: (name: string) => void
}

// Consistent world-unit scale across all features
const MAP_SPAN = 12.0

function getProjectionParams(b: Bounds) {
  const centerLon = (b.minLon + b.maxLon) / 2
  const centerLat = (b.minLat + b.maxLat) / 2
  const scale = MAP_SPAN / (b.maxLon - b.minLon)
  const cosLat = Math.cos((centerLat * Math.PI) / 180)
  return { centerLon, centerLat, scale, cosLat }
}

function project3D(lon: number, lat: number, b: Bounds): [number, number] {
  const { centerLon, centerLat, scale, cosLat } = getProjectionParams(b)
  const x = (lon - centerLon) * scale * cosLat
  const z = -(lat - centerLat) * scale
  return [x, z]
}

// Known hotspots where 3D vertical risk pillars rise up from states/districts
const PILLAR_LOCATIONS = [
  { stateName: 'Uttar Pradesh', district: 'Lucknow/Central UP', lon: 80.57, lat: 26.92 },
  { stateName: 'Uttar Pradesh', district: 'Sonbhadra', lon: 83.00, lat: 24.68 },
  { stateName: 'Bihar', district: 'Patna/Gaya', lon: 85.61, lat: 25.68 },
  { stateName: 'Delhi', district: 'Delhi NCR', lon: 77.13, lat: 28.64 },
  { stateName: 'Punjab', district: 'Ludhiana', lon: 75.42, lat: 30.85 },
  { stateName: 'Haryana', district: 'Rohtak', lon: 76.34, lat: 29.21 },
  { stateName: 'Rajasthan', district: 'Barmer/Jaipur', lon: 73.86, lat: 26.58 },
  { stateName: 'Madhya Pradesh', district: 'Bhopal', lon: 78.30, lat: 23.54 },
  { stateName: 'Maharashtra', district: 'Aurangabad', lon: 76.11, lat: 19.45 },
  { stateName: 'Maharashtra', district: 'Gadchiroli', lon: 80.00, lat: 20.18 },
  { stateName: 'Jharkhand', district: 'Latehar/Ranchi', lon: 85.56, lat: 23.66 },
  { stateName: 'Odisha', district: 'Bhubaneswar', lon: 84.43, lat: 20.51 },
  { stateName: 'Odisha', district: 'Koraput', lon: 82.71, lat: 18.81 },
  { stateName: 'Chhattisgarh', district: 'Raipur', lon: 82.04, lat: 21.27 },
  { stateName: 'Telangana', district: 'Hyderabad', lon: 79.01, lat: 17.80 },
  { stateName: 'Andhra Pradesh', district: 'Amaravati', lon: 79.97, lat: 15.75 },
  { stateName: 'Karnataka', district: 'Bengaluru/Central', lon: 76.17, lat: 14.71 },
  { stateName: 'Tamil Nadu', district: 'Madurai/Trichy', lon: 78.41, lat: 11.02 },
  { stateName: 'Kerala', district: 'Kochi', lon: 76.41, lat: 10.44 },
  { stateName: 'Gujarat', district: 'Ahmedabad', lon: 71.60, lat: 22.69 },
  { stateName: 'West Bengal', district: 'Kolkata/Burdwan', lon: 87.95, lat: 23.91 },
  { stateName: 'Arunachal Pradesh', district: 'Itanagar', lon: 94.68, lat: 28.03 },
  { stateName: 'Assam', district: 'Guwahati', lon: 92.82, lat: 26.35 },
]

// Custom styling and tier for state labels to avoid collisions in dense regions
const LABEL_STYLE_TIERS: Record<string, 'normal' | 'small' | 'tiny' | 'hide'> = {
  // Dense Northeastern states: reduced font to avoid illegible overlap
  Nagaland: 'tiny',
  Manipur: 'tiny',
  Mizoram: 'tiny',
  Tripura: 'tiny',
  Sikkim: 'tiny',
  Goa: 'tiny',
  'Arunachal Pradesh': 'small',
  Assam: 'small',
  Meghalaya: 'small',
  // Tiny union territories that don't need text collision
  Chandigarh: 'hide',
  'Dadra and Nagar Haveli': 'hide',
  'Daman and Diu': 'hide',
  Puducherry: 'hide',
}

function PillarMesh({
  lon,
  lat,
  stateName,
  valueFor,
  bounds,
}: {
  lon: number
  lat: number
  stateName: string
  valueFor: (name: string) => number
  bounds: Bounds
}) {
  const [x, z] = project3D(lon, lat, bounds)
  const val = valueFor(stateName)
  const height = Math.max(0.9, val * 4.2)
  const color = riskColor(val)
  const radius = 0.11

  return (
    <group position={[x, 0.28, z]}>
      {/* 3D Vertical Pillar Cylinder */}
      <mesh position={[0, height / 2, 0]} castShadow>
        <cylinderGeometry args={[radius, radius, height, 16]} />
        <meshStandardMaterial
          color={color}
          roughness={0.15}
          metalness={0.2}
          emissive={color}
          emissiveIntensity={0.35}
        />
      </mesh>

      {/* Glowing Top Cap */}
      <mesh position={[0, height, 0]}>
        <sphereGeometry args={[radius * 1.3, 16, 16]} />
        <meshStandardMaterial
          color="#ffffff"
          roughness={0.1}
          metalness={0.4}
          emissive={color}
          emissiveIntensity={0.7}
        />
      </mesh>
    </group>
  )
}

function StateMesh({
  feature,
  bounds,
  value,
  name,
  onClick,
}: {
  feature: any
  bounds: Bounds
  value: number
  name: string
  onClick: () => void
}) {
  const geometries = useMemo(() => {
    const polys = featurePolygons(feature)
    if (!polys || !polys.length) return []

    const geos: THREE.ExtrudeGeometry[] = []

    for (const poly of polys) {
      if (!poly || !poly.length) continue
      const rawOuter = poly[0]
      if (!rawOuter || rawOuter.length < 3) continue

      // Use raw rings directly without decimation to ensure seamless contiguous borders
      const shape = new THREE.Shape()

      rawOuter.forEach(([lon, lat], i) => {
        const [x, z] = project3D(lon, lat, bounds)
        // Shape coordinates (x, -z) map to 3D world (x, depth, z) after rotateX(-Math.PI / 2)
        if (i === 0) shape.moveTo(x, -z)
        else shape.lineTo(x, -z)
      })
      shape.closePath()

      // Process inner hole rings (if any)
      poly.slice(1).forEach((rawHole) => {
        if (!rawHole || rawHole.length < 3) return
        const holePath = new THREE.Path()
        rawHole.forEach(([lon, lat], i) => {
          const [x, z] = project3D(lon, lat, bounds)
          if (i === 0) holePath.moveTo(x, -z)
          else holePath.lineTo(x, -z)
        })
        holePath.closePath()
        shape.holes.push(holePath)
      })

      const depth = Math.max(0.18, 0.12 + value * 0.45)
      const geo = new THREE.ExtrudeGeometry(shape, {
        depth,
        bevelEnabled: true,
        bevelSegments: 1,
        bevelSize: 0.012,
        bevelThickness: 0.012,
      })
      geo.rotateX(-Math.PI / 2)
      geos.push(geo)
    }

    return geos
  }, [feature, bounds, value])

  if (!geometries.length) return null

  const color = riskColor(value)

  return (
    <group
      name={name}
      onClick={(e) => {
        e.stopPropagation()
        onClick()
      }}
    >
      {geometries.map((geo, idx) => (
        <mesh key={idx} geometry={geo} receiveShadow castShadow>
          <meshStandardMaterial color={color} roughness={0.35} metalness={0.12} />
        </mesh>
      ))}
    </group>
  )
}

function SeaWaterPlane() {
  return (
    <mesh position={[0, -0.02, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
      <planeGeometry args={[40, 40]} />
      <meshStandardMaterial color="#132b38" roughness={0.7} metalness={0.25} />
    </mesh>
  )
}

function Lighting() {
  const dirLightRef = useRef<THREE.DirectionalLight>(null)
  useFrame(({ clock }) => {
    if (dirLightRef.current) {
      dirLightRef.current.position.x = 8 + Math.sin(clock.getElapsedTime() * 0.15) * 0.4
    }
  })

  return (
    <>
      <ambientLight intensity={1.6} color="#f0f9ff" />
      <directionalLight
        ref={dirLightRef}
        position={[8, 18, 10]}
        intensity={1.8}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <directionalLight position={[-10, 12, -8]} intensity={0.7} color="#bae6fd" />
    </>
  )
}

export default function Map3D({ geoData, bounds, valueFor, onSelectDistrict }: Props) {
  // Precompute true centroids for all features in the GeoJSON
  const stateLabels = useMemo(() => {
    return geoData.features
      .map((f: any) => {
        const rawName = getStateName(f)
        const name = normalizeStateName(rawName)
        const tier = LABEL_STYLE_TIERS[name] ?? 'normal'
        if (tier === 'hide') return null

        const [lon, lat] = getFeatureCentroidLonLat(f)
        const [px, pz] = project3D(lon, lat, bounds)
        const isDelhi = name === 'Delhi'

        return {
          name,
          lon,
          lat,
          px,
          pz,
          tier,
          isDelhi,
        }
      })
      .filter(Boolean) as Array<{
      name: string
      lon: number
      lat: number
      px: number
      pz: number
      tier: 'normal' | 'small' | 'tiny'
      isDelhi: boolean
    }>
  }, [geoData, bounds])

  return (
    <div className="relative h-[500px] w-full overflow-hidden bg-[#112330]">
      <Canvas
        camera={{ position: [0, 11.5, 11.0], fov: 42 }}
        shadows
        className="cursor-grab active:cursor-grabbing"
      >
        <Lighting />

        {/* Root Group centered on origin (0, 0, 0) */}
        <group position={[0, 0, 0]}>
          <SeaWaterPlane />

          {/* 3D Contiguous State Meshes */}
          {geoData.features.map((f: any, i: number) => {
            const rawName = getStateName(f) || `state-${i}`
            const name = normalizeStateName(rawName)
            return (
              <StateMesh
                key={rawName + i}
                feature={f}
                bounds={bounds}
                value={valueFor(name)}
                name={name}
                onClick={() => onSelectDistrict?.(name)}
              />
            )
          })}

          {/* 3D Vertical Risk Pillars */}
          {PILLAR_LOCATIONS.map((p, idx) => (
            <PillarMesh
              key={idx}
              lon={p.lon}
              lat={p.lat}
              stateName={p.stateName}
              valueFor={valueFor}
              bounds={bounds}
            />
          ))}

          {/* State Names Overlaid at True Mathematical Centroids */}
          {stateLabels.map((item) => (
            <Html
              key={item.name}
              position={[item.px, 0.45, item.pz]}
              center
              distanceFactor={18}
              className="pointer-events-none select-none"
            >
              <div className="flex items-center gap-0.5 whitespace-nowrap">
                {item.isDelhi && (
                  <span className="h-1.5 w-1.5 rounded-full bg-red-500 shadow-[0_0_6px_#ef4444]" />
                )}
                <span
                  className={`${
                    item.tier === 'tiny'
                      ? 'text-[6.5px] font-semibold text-slate-300'
                      : item.tier === 'small'
                      ? 'text-[7.5px] font-semibold text-slate-200'
                      : 'text-[8.5px] font-bold text-slate-100'
                  } ${item.isDelhi ? 'text-red-400 font-extrabold' : ''}`}
                  style={{
                    textShadow:
                      '0 1px 2px rgba(0,0,0,0.95), 0 0 3px rgba(0,0,0,0.9), 0 0 6px rgba(0,0,0,0.85)',
                  }}
                >
                  {item.name}
                </span>
              </div>
            </Html>
          ))}

          {/* Oceanic Geographic Labels in 3D scene */}
          <Html
            position={[-5.8, 0.05, 1.8]}
            center
            distanceFactor={22}
            className="pointer-events-none select-none"
          >
            <span
              className="font-serif text-sm italic tracking-widest text-[#94a3b8]/60"
              style={{ textShadow: '0 1px 4px rgba(0,0,0,0.9)' }}
            >
              Arabian Sea
            </span>
          </Html>
          <Html
            position={[5.6, 0.05, 1.6]}
            center
            distanceFactor={22}
            className="pointer-events-none select-none"
          >
            <span
              className="font-serif text-sm italic tracking-widest text-[#94a3b8]/60"
              style={{ textShadow: '0 1px 4px rgba(0,0,0,0.9)' }}
            >
              Bay of Bengal
            </span>
          </Html>
          <Html
            position={[0, 0.05, 7.2]}
            center
            distanceFactor={22}
            className="pointer-events-none select-none"
          >
            <span
              className="font-serif text-sm italic tracking-widest text-[#94a3b8]/60"
              style={{ textShadow: '0 1px 4px rgba(0,0,0,0.9)' }}
            >
              Indian Ocean
            </span>
          </Html>
        </group>

        <OrbitControls
          enablePan={true}
          panSpeed={0.8}
          rotateSpeed={0.7}
          zoomSpeed={0.8}
          minDistance={6}
          maxDistance={22}
          maxPolarAngle={Math.PI / 2.15}
          target={[0, 0, 0]}
        />
      </Canvas>
    </div>
  )
}
