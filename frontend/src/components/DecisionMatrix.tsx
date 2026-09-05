import { useMemo } from 'react'
import { riskConfidencePoints } from '../data/mockData'

const QUADRANT_COLORS: Record<string, string> = {
  Investigate: '#dc2626',
  Remediate: '#ea580c',
  Monitor: '#84cc16',
  Clear: '#16a34a',
}

export default function DecisionMatrix() {
  const points = useMemo(() => riskConfidencePoints(), [])

  // Optimized viewBox coordinates
  const svgWidth = 360
  const svgHeight = 165
  const margin = { top: 16, right: 18, bottom: 28, left: 34 }

  const plotWidth = svgWidth - margin.left - margin.right
  const plotHeight = svgHeight - margin.top - margin.bottom

  const toSvgX = (r: number) => margin.left + r * plotWidth
  const toSvgY = (c: number) => margin.top + (1 - c) * plotHeight

  return (
    <div className="flex flex-col justify-between rounded-xl border border-[#d8cbb0] bg-[#fbf9f4] p-3.5 shadow-xs">
      <div className="mb-1 flex items-center justify-between">
        <h3 className="font-serif text-xs font-bold text-[#0b1a2d]">Risk vs Confidence</h3>
      </div>

      <div className="relative w-full">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="h-auto w-full"
        >
          {/* Chart Background */}
          <rect
            x={margin.left}
            y={margin.top}
            width={plotWidth}
            height={plotHeight}
            fill="#ffffff"
            stroke="#e2d5bd"
            strokeWidth="0.8"
          />

          {/* Crosshairs at 0.5 */}
          <line
            x1={toSvgX(0.5)}
            y1={margin.top}
            x2={toSvgX(0.5)}
            y2={margin.top + plotHeight}
            stroke="#e2e8f0"
            strokeWidth="1"
            strokeDasharray="2 2"
          />
          <line
            x1={margin.left}
            y1={toSvgY(0.5)}
            x2={margin.left + plotWidth}
            y2={toSvgY(0.5)}
            stroke="#e2e8f0"
            strokeWidth="1"
            strokeDasharray="2 2"
          />

          {/* Quadrant Text Labels matching screenshot */}
          <text
            x={toSvgX(0.25)}
            y={toSvgY(0.9)}
            textAnchor="middle"
            fontSize="9"
            fontWeight="600"
            fill="#15803d"
          >
            Monitor
          </text>
          <text
            x={toSvgX(0.75)}
            y={toSvgY(0.9)}
            textAnchor="middle"
            fontSize="9"
            fontWeight="600"
            fill="#b91c1c"
          >
            Investigate
          </text>
          <text
            x={toSvgX(0.25)}
            y={toSvgY(0.12)}
            textAnchor="middle"
            fontSize="9"
            fontWeight="600"
            fill="#16a34a"
          >
            Clear
          </text>
          <text
            x={toSvgX(0.75)}
            y={toSvgY(0.12)}
            textAnchor="middle"
            fontSize="9"
            fontWeight="600"
            fill="#c2410c"
          >
            Remediate
          </text>

          {/* Scatter Points */}
          {points.map((p, idx) => (
            <circle
              key={idx}
              cx={toSvgX(p.risk)}
              cy={toSvgY(p.confidence)}
              r="1.4"
              fill={QUADRANT_COLORS[p.decision] || '#64748b'}
              opacity={0.8}
            />
          ))}

          {/* X Axis ticks: 0, 0.5, 1.0 */}
          <text
            x={toSvgX(0)}
            y={margin.top + plotHeight + 11}
            textAnchor="middle"
            fontSize="8"
            fill="#64748b"
          >
            0
          </text>
          <text
            x={toSvgX(0.5)}
            y={margin.top + plotHeight + 11}
            textAnchor="middle"
            fontSize="8"
            fill="#64748b"
          >
            0.5
          </text>
          <text
            x={toSvgX(1)}
            y={margin.top + plotHeight + 11}
            textAnchor="middle"
            fontSize="8"
            fill="#64748b"
          >
            1.0
          </text>
          <text
            x={margin.left + plotWidth / 2}
            y={margin.top + plotHeight + 22}
            textAnchor="middle"
            fontSize="8.5"
            fontWeight="500"
            fill="#475569"
          >
            Risk Score (R)
          </text>

          {/* Y Axis ticks: 0.0, 0.5, 1.0 */}
          <text
            x={margin.left - 4}
            y={toSvgY(0) + 3}
            textAnchor="end"
            fontSize="8"
            fill="#64748b"
          >
            0
          </text>
          <text
            x={margin.left - 4}
            y={toSvgY(0.5) + 3}
            textAnchor="end"
            fontSize="8"
            fill="#64748b"
          >
            0.5
          </text>
          <text
            x={margin.left - 4}
            y={toSvgY(1) + 3}
            textAnchor="end"
            fontSize="8"
            fill="#64748b"
          >
            1.0
          </text>
          <text
            transform={`rotate(-90 ${margin.left - 20} ${margin.top + plotHeight / 2})`}
            x={margin.left - 20}
            y={margin.top + plotHeight / 2}
            textAnchor="middle"
            fontSize="8.5"
            fontWeight="500"
            fill="#475569"
          >
            Confidence (C)
          </text>
        </svg>
      </div>
    </div>
  )
}
