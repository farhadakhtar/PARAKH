import { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { featurePolygons, getStateName } from '../data/indiaGeo'
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

const SCALE = 8 // world-unit span for the longer map axis

function project(lon: number, lat: number, b: Bounds): [number, number] {
  const spanLon = b.maxLon - b.minLon
  const spanLat = b.maxLat - b.minLat
  const span = Math.max(spanLon, spanLat)
  const x = ((lon - b.minLon) / span) * SCALE
  const z = ((lat - b.minLat) / span) * SCALE
  return [x, z]
}

// Decimates vertices for complex polygons to ensure smooth 60 FPS Three.js performance
function decimateRing(ring: Array<[number, number]>, step = 2): Array<[number, number]> {
  if (ring.length <= 16) return ring
  const res: Array<[number, number]> = []
  for (let i = 0; i < ring.length; i += step) {
    res.push(ring[i])
  }
  // ensure closed ring
  if (res.length > 0 && ring.length > 0) {
    res[res.length - 1] = ring[ring.length - 1]
  }
  return res
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

      // Process holes
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

      const depth = Math.max(0.08, 0.05 + value * 0.7)
      const geo = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false })
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
        <mesh key={idx} geometry={geo}>
          <meshStandardMaterial
            color={color}
            roughness={0.5}
            metalness={0.1}
          />
        </mesh>
      ))}
    </group>
  )
}

export default function Map3D({ geoData, bounds, valueFor, onSelectDistrict }: Props) {
  return (
    <div className="h-[560px] w-full overflow-hidden rounded-xl border border-navy/10 bg-gradient-to-b from-parchment to-parchment-deep shadow-inner">
      <Canvas camera={{ position: [0, 8, 8], fov: 42 }} shadows>
        <ambientLight intensity={0.8} />
        <directionalLight position={[5, 12, 6]} intensity={1.2} castShadow />
        <directionalLight position={[-5, 8, -4]} intensity={0.4} />
        <group position={[-SCALE / 2, 0, -SCALE / 2]}>
          {geoData.features.map((f: any, i: number) => {
            const name = getStateName(f) || f.properties?.NAME_1 || f.properties?.name || f.properties?.st_nm || `state-${i}`
            return (
              <StateMesh
                key={name + i}
                feature={f}
                bounds={bounds}
                value={valueFor(name)}
                name={name}
                onClick={() => onSelectDistrict?.(name)}
              />
            )
          })}
        </group>
        <OrbitControls
          enablePan={false}
          minDistance={4}
          maxDistance={15}
          maxPolarAngle={Math.PI / 2.15}
        />
      </Canvas>
    </div>
  )
}
