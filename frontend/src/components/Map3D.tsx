import { useMemo, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import * as THREE from 'three'
import { featurePolygons, getStateName, normalizeStateName } from '../data/indiaGeo'
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

// Larger scale so the sub-continent expands across the ocean canvas
const SCALE = 14.2

function project(lon: number, lat: number, b: Bounds): [number, number] {
  const spanLon = b.maxLon - b.minLon
  const spanLat = b.maxLat - b.minLat
  const span = Math.max(spanLon, spanLat)
  const x = ((lon - b.minLon) / span) * SCALE
  const z = -((lat - b.minLat) / span) * SCALE
  return [x, z]
}

function decimateRing(ring: Array<[number, number]>, step = 2): Array<[number, number]> {
  if (ring.length <= 16) return ring
  const res: Array<[number, number]> = []
  for (let i = 0; i < ring.length; i += step) {
    res.push(ring[i])
  }
  if (res.length > 0 && ring.length > 0) {
    res[res.length - 1] = ring[ring.length - 1]
  }
  return res
}

// 3D vertical risk pillar bars rising up from high-risk zones
const RISK_PILLARS = [
  // Uttar Pradesh / Delhi / Bihar dense high cluster
  { lon: 80.94, lat: 26.85, height: 3.6, color: '#dc2626' },
  { lon: 81.90, lat: 26.20, height: 4.4, color: '#b91c1c' },
  { lon: 83.00, lat: 25.10, height: 5.0, color: '#991b1b' }, // Sonbhadra/Varanasi
  { lon: 85.31, lat: 25.10, height: 3.4, color: '#dc2626' }, // Bihar
  { lon: 77.21, lat: 28.61, height: 3.2, color: '#ef4444' }, // Delhi
  { lon: 75.34, lat: 31.15, height: 2.8, color: '#e11d48' }, // Punjab
  { lon: 76.09, lat: 29.06, height: 2.2, color: '#ea580c' }, // Haryana
  { lon: 73.80, lat: 26.50, height: 2.2, color: '#ea580c' }, // Rajasthan
  { lon: 77.80, lat: 23.30, height: 2.0, color: '#f59e0b' }, // Madhya Pradesh
  { lon: 75.71, lat: 19.75, height: 2.6, color: '#ea580c' }, // Maharashtra
  { lon: 80.00, lat: 20.10, height: 3.5, color: '#dc2626' }, // Gadchiroli
  { lon: 85.28, lat: 23.61, height: 2.9, color: '#ea580c' }, // Jharkhand (Latehar)
  { lon: 84.46, lat: 20.95, height: 2.5, color: '#ea580c' }, // Odisha
  { lon: 82.70, lat: 18.80, height: 3.1, color: '#dc2626' }, // Koraput
  { lon: 81.87, lat: 21.28, height: 1.9, color: '#f59e0b' }, // Chhattisgarh
  { lon: 79.02, lat: 18.11, height: 1.8, color: '#84cc16' }, // Telangana
  { lon: 79.74, lat: 15.91, height: 1.9, color: '#eab308' }, // Andhra Pradesh
  { lon: 75.71, lat: 15.32, height: 1.6, color: '#84cc16' }, // Karnataka
  { lon: 78.66, lat: 11.13, height: 1.5, color: '#16a34a' }, // Tamil Nadu
  { lon: 76.27, lat: 10.85, height: 1.4, color: '#16a34a' }, // Kerala
  { lon: 71.19, lat: 22.26, height: 1.6, color: '#eab308' }, // Gujarat
  { lon: 87.85, lat: 23.40, height: 1.8, color: '#f59e0b' }, // West Bengal
  { lon: 94.73, lat: 28.22, height: 1.2, color: '#16a34a' }, // Arunachal
  { lon: 92.94, lat: 26.20, height: 1.2, color: '#84cc16' }, // Assam
]

// Prominent regional labels with exact coordinates
const STATE_LABELS = [
  { name: 'Jammu & Kashmir', lon: 74.80, lat: 34.00, y: 0.5 },
  { name: 'Ladakh', lon: 78.20, lat: 34.50, y: 0.5 },
  { name: 'Himachal Pradesh', lon: 77.17, lat: 31.90, y: 0.5 },
  { name: 'Punjab', lon: 75.10, lat: 31.15, y: 0.5 },
  { name: 'Haryana', lon: 76.20, lat: 29.30, y: 0.5 },
  { name: 'Delhi', lon: 77.21, lat: 28.61, y: 0.6, isPin: true },
  { name: 'Rajasthan', lon: 72.80, lat: 26.50, y: 0.5 },
  { name: 'Uttar Pradesh', lon: 80.94, lat: 27.20, y: 0.5 },
  { name: 'Bihar', lon: 85.31, lat: 25.50, y: 0.5 },
  { name: 'Madhya Pradesh', lon: 77.80, lat: 23.30, y: 0.5 },
  { name: 'Gujarat', lon: 70.80, lat: 22.26, y: 0.5 },
  { name: 'Maharashtra', lon: 75.40, lat: 19.60, y: 0.5 },
  { name: 'Chhattisgarh', lon: 81.87, lat: 21.28, y: 0.5 },
  { name: 'Jharkhand', lon: 85.28, lat: 23.61, y: 0.5 },
  { name: 'Odisha', lon: 84.46, lat: 20.80, y: 0.5 },
  { name: 'West Bengal', lon: 87.85, lat: 23.40, y: 0.5 },
  { name: 'Telangana', lon: 79.02, lat: 18.00, y: 0.5 },
  { name: 'Andhra Pradesh', lon: 79.74, lat: 15.60, y: 0.5 },
  { name: 'Karnataka', lon: 75.60, lat: 14.80, y: 0.5 },
  { name: 'Goa', lon: 73.90, lat: 15.30, y: 0.4 },
  { name: 'Tamil Nadu', lon: 78.66, lat: 10.80, y: 0.5 },
  { name: 'Kerala', lon: 76.27, lat: 10.40, y: 0.5 },
  { name: 'Arunachal Pradesh', lon: 94.73, lat: 28.22, y: 0.5 },
  { name: 'Meghalaya', lon: 91.37, lat: 25.47, y: 0.5 },
  { name: 'Nagaland', lon: 94.56, lat: 26.16, y: 0.5 },
  { name: 'Manipur', lon: 93.91, lat: 24.66, y: 0.5 },
  { name: 'Mizoram', lon: 92.94, lat: 23.16, y: 0.5 },
  { name: 'Tripura', lon: 91.80, lat: 23.80, y: 0.5 },
  { name: 'Lakshadweep', lon: 72.40, lat: 10.57, y: 0.3 },
  { name: 'Andaman & Nicobar Islands', lon: 92.66, lat: 11.74, y: 0.3 },
]

function PillarMesh({
  lon,
  lat,
  height,
  color,
  bounds,
}: {
  lon: number
  lat: number
  height: number
  color: string
  bounds: Bounds
}) {
  const [x, z] = project(lon, lat, bounds)
  const radius = 0.12

  return (
    <group position={[x, 0.3, z]}>
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

      {/* Glowing Cap */}
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

      const outer = decimateRing(rawOuter, 2)
      const shape = new THREE.Shape()

      outer.forEach(([lon, lat], i) => {
        const [x, z] = project(lon, lat, bounds)
        if (i === 0) shape.moveTo(x, z)
        else shape.lineTo(x, z)
      })

      poly.slice(1).forEach((rawHole) => {
        if (!rawHole || rawHole.length < 3) return
        const holeRing = decimateRing(rawHole, 2)
        const holePath = new THREE.Path()
        holeRing.forEach(([lon, lat], i) => {
          const [x, z] = project(lon, lat, bounds)
          if (i === 0) holePath.moveTo(x, z)
          else holePath.lineTo(x, z)
        })
        shape.holes.push(holePath)
      })

      const depth = Math.max(0.18, 0.12 + value * 0.5)
      const geo = new THREE.ExtrudeGeometry(shape, {
        depth,
        bevelEnabled: true,
        bevelSegments: 1,
        bevelSize: 0.016,
        bevelThickness: 0.016,
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
        <mesh key={idx} geometry={geo} receiveShadow>
          <meshStandardMaterial
            color={color}
            roughness={0.3}
            metalness={0.1}
          />
        </mesh>
      ))}
    </group>
  )
}

function SeaWaterPlane() {
  return (
    <mesh position={[0, -0.05, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
      <planeGeometry args={[45, 45]} />
      <meshStandardMaterial color="#142b38" roughness={0.65} metalness={0.25} />
    </mesh>
  )
}

function Lighting() {
  const dirLightRef = useRef<THREE.DirectionalLight>(null)
  useFrame(({ clock }) => {
    if (dirLightRef.current) {
      dirLightRef.current.position.x = 8 + Math.sin(clock.getElapsedTime() * 0.15) * 0.5
    }
  })

  return (
    <>
      <ambientLight intensity={1.5} color="#f0f9ff" />
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
  const [centerX, centerZ] = project(
    (bounds.minLon + bounds.maxLon) / 2,
    (bounds.minLat + bounds.maxLat) / 2,
    bounds
  )

  return (
    <div className="relative h-[500px] w-full overflow-hidden bg-[#112330]">
      <Canvas
        camera={{ position: [0, 12.5, 12.5], fov: 38 }}
        shadows
        className="cursor-grab active:cursor-grabbing"
      >
        <Lighting />

        <group position={[-centerX, 0, -centerZ]}>
          <SeaWaterPlane />

          {/* 3D State Meshes */}
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
          {RISK_PILLARS.map((p, idx) => (
            <PillarMesh
              key={idx}
              lon={p.lon}
              lat={p.lat}
              height={p.height}
              color={p.color}
              bounds={bounds}
            />
          ))}

          {/* State Names Overlaid in 3D Scene */}
          {STATE_LABELS.map((item) => {
            const [px, pz] = project(item.lon, item.lat, bounds)
            return (
              <Html
                key={item.name}
                position={[px, item.y, pz]}
                center
                distanceFactor={19}
                className="pointer-events-none select-none"
              >
                <div className="flex items-center gap-1 whitespace-nowrap">
                  {item.isPin && (
                    <span className="h-1.5 w-1.5 rounded-full bg-red-500 shadow-[0_0_6px_#ef4444]" />
                  )}
                  <span
                    className={`text-[8.5px] font-bold ${
                      item.isPin ? 'text-red-400 font-extrabold' : 'text-slate-100'
                    }`}
                    style={{
                      textShadow:
                        '0 1px 3px rgba(0,0,0,0.95), 0 0 3px rgba(0,0,0,0.9), 0 0 6px rgba(0,0,0,0.85)',
                    }}
                  >
                    {item.name}
                  </span>
                </div>
              </Html>
            )
          })}

          {/* Oceanic Geographic Labels in 3D scene */}
          <Html position={[-6.0, 0.05, 2.5]} center distanceFactor={22} className="pointer-events-none select-none">
            <span
              className="font-serif text-sm italic tracking-widest text-[#94a3b8]/60"
              style={{ textShadow: '0 1px 4px rgba(0,0,0,0.9)' }}
            >
              Arabian Sea
            </span>
          </Html>
          <Html position={[5.8, 0.05, 2.2]} center distanceFactor={22} className="pointer-events-none select-none">
            <span
              className="font-serif text-sm italic tracking-widest text-[#94a3b8]/60"
              style={{ textShadow: '0 1px 4px rgba(0,0,0,0.9)' }}
            >
              Bay of Bengal
            </span>
          </Html>
          <Html position={[0.5, 0.05, 7.5]} center distanceFactor={22} className="pointer-events-none select-none">
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
          minDistance={7}
          maxDistance={22}
          maxPolarAngle={Math.PI / 2.15}
          target={[0, 0.6, 0]}
        />
      </Canvas>
    </div>
  )
}
