import React from 'react'

/** Lion Capital of Ashoka (National Emblem of India) in stylized gold */
export const LionCapitalEmblem: React.FC<{ className?: string; size?: number; color?: string }> = ({
  className = '',
  size = 48,
  color = '#c9a227',
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 100 125"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
  >
    {/* Stylized Four Lions Capital */}
    <g fill={color}>
      {/* Central Lion Head */}
      <path d="M50 8C43 8 38 12 37 17C35 15 31 16 30 19C28 22 30 26 31 28C28 30 27 34 29 38C30 40 33 41 36 41C38 45 42 49 50 49C58 49 62 45 64 41C67 41 70 40 71 38C73 34 72 30 69 28C70 26 72 22 70 19C69 16 65 15 63 17C62 12 57 8 50 8Z" opacity="0.95" />
      {/* Lion Manes & Facial Definition */}
      <path d="M50 14C46 14 43 17 43 20C43 23 46 25 50 25C54 25 57 23 57 20C57 17 54 14 50 14Z" fill="#1a1103" opacity="0.3" />
      <circle cx="46.5" cy="20" r="1.5" fill="#f5efe0" />
      <circle cx="53.5" cy="20" r="1.5" fill="#f5efe0" />
      <path d="M48 23L50 25L52 23H48Z" fill="#f5efe0" />
      <path d="M45 28C47 30 53 30 55 28" stroke="#f5efe0" strokeWidth="1.2" strokeLinecap="round" />
      
      {/* Left Lion Profile */}
      <path d="M30 22C24 24 20 29 20 35C20 42 25 47 31 48C30 44 31 38 34 35C33 30 33 25 30 22Z" opacity="0.85" />
      {/* Right Lion Profile */}
      <path d="M70 22C76 24 80 29 80 35C80 42 75 47 69 48C70 44 69 38 66 35C67 30 67 25 70 22Z" opacity="0.85" />

      {/* Mane details & Paws */}
      <path d="M34 44C30 48 30 56 34 60C38 64 43 66 50 66C57 66 62 64 66 60C70 56 70 48 66 44C62 48 58 50 50 50C42 50 38 48 34 44Z" />
      <path d="M42 55C40 60 41 68 44 72H56C59 68 60 60 58 55C55 58 52 59 50 59C48 59 45 58 42 55Z" fill="#e6cd7a" />
      
      {/* Abacus frieze */}
      <rect x="22" y="74" width="56" height="10" rx="2" />
      {/* Ashoka Chakra in frieze */}
      <circle cx="50" cy="79" r="4" fill="#0d1b2a" />
      <circle cx="50" cy="79" r="3.2" stroke="#e6cd7a" strokeWidth="0.6" fill="none" />
      <circle cx="50" cy="79" r="0.8" fill="#e6cd7a" />
      {/* Flanking Animals on Frieze (Bull & Horse silhouettes) */}
      <ellipse cx="32" cy="79" rx="3.5" ry="2" fill="#0d1b2a" />
      <ellipse cx="68" cy="79" rx="3.5" ry="2" fill="#0d1b2a" />

      {/* Inverted Lotus Bell Base */}
      <path d="M26 86C30 92 40 96 50 96C60 96 70 92 74 86C72 85 28 85 26 86Z" />
      <path d="M30 87C34 94 42 98 50 98C58 98 66 94 70 87" stroke="#e6cd7a" strokeWidth="1" fill="none" />
      
      {/* Plinth Base */}
      <rect x="18" y="98" width="64" height="4" rx="1" />
      <rect x="14" y="103" width="72" height="5" rx="1.5" />
    </g>
    {/* Satyamev Jayate in Devanagari */}
    <text
      x="50"
      y="119"
      textAnchor="middle"
      fontSize="7.5"
      fontWeight="700"
      fill={color}
      fontFamily="'Playfair Display', serif"
      letterSpacing="0.8"
    >
      सत्यमेव जयते
    </text>
  </svg>
)

/** Sansad Bhavan (Old Indian Parliament) fine line-art illustration */
export const SansadBhavanIllustration: React.FC<{ className?: string; color?: string }> = ({
  className = '',
  color = '#c9a227',
}) => (
  <svg
    viewBox="0 0 200 90"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
  >
    {/* Flag on Central Mast */}
    <line x1="100" y1="8" x2="100" y2="28" stroke={color} strokeWidth="1.5" />
    <path d="M100 8L112 11.5L100 15V8Z" fill="#ff9933" />
    <circle cx="100" cy="7" r="1.5" fill={color} />

    {/* Central Dome */}
    <path d="M84 34C84 25 91 22 100 22C109 22 116 25 116 34H84Z" fill={color} fillOpacity="0.25" stroke={color} strokeWidth="1.2" />
    <rect x="80" y="34" width="40" height="4" fill={color} stroke={color} strokeWidth="1" />

    {/* Upper Drum / Attique */}
    <rect x="40" y="38" width="120" height="8" rx="1" stroke={color} strokeWidth="1.2" fill={color} fillOpacity="0.1" />
    {/* Circular Colonnade Pillars (The 144 columns of circular colonnade) */}
    <g stroke={color} strokeWidth="1.2">
      {Array.from({ length: 25 }).map((_, i) => (
        <line key={i} x1={44 + i * 4.6} y1="46" x2={44 + i * 4.6} y2="68" strokeOpacity="0.75" />
      ))}
    </g>

    {/* Entablature above colonnade */}
    <rect x="36" y="44" width="128" height="3" fill={color} stroke={color} strokeWidth="0.8" />
    {/* Base Podium & Steps */}
    <rect x="32" y="68" width="136" height="5" fill={color} fillOpacity="0.2" stroke={color} strokeWidth="1.2" />
    <rect x="24" y="73" width="152" height="4" fill={color} stroke={color} strokeWidth="1" />
    <rect x="16" y="77" width="168" height="4" fill={color} stroke={color} strokeWidth="1" />
    <line x1="8" y1="82" x2="192" y2="82" stroke={color} strokeWidth="1.5" strokeOpacity="0.6" />
  </svg>
)

/** Digital India official emblem vector */
export const DigitalIndiaLogo: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`flex items-center gap-2.5 ${className}`}>
    {/* Stylized Tricolor Flame swirl */}
    <svg width="32" height="32" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        d="M24 4C14 4 6 12 6 22C6 32 14 40 24 40C30 40 35 37 38 32C36 33 33 34 30 34C21 34 14 27 14 18C14 11 18 6 24 4Z"
        fill="#0072bc"
      />
      <path
        d="M24 10C18 10 13 15 13 21C13 27 18 32 24 32C28 32 31 30 33 27C31 28 29 28 27 28C22 28 18 24 18 19C18 14 21 11 24 10Z"
        fill="#f37023"
      />
      <path
        d="M24 16C21 16 19 18 19 21C19 24 21 26 24 26C26 26 28 25 29 23C27 24 26 24 25 24C23 24 21 22 21 20C21 18 22 17 24 16Z"
        fill="#39b54a"
      />
      <circle cx="28" cy="18" r="3" fill="#f7931e" />
    </svg>
    <div className="flex flex-col leading-tight">
      <span className="font-sans text-xs font-bold tracking-tight text-white">Digital India</span>
      <span className="text-[8px] font-medium tracking-wide text-slate-400">Power To Empower</span>
    </div>
  </div>
)

/** Dynamic Indian Tricolor Swoosh Wave */
export const TricolorWave: React.FC<{ className?: string }> = ({ className = '' }) => (
  <svg
    viewBox="0 0 160 30"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={`h-7 w-28 overflow-visible ${className}`}
  >
    {/* Saffron band */}
    <path
      d="M0 12C30 6 60 2 100 8C130 13 145 7 160 3"
      stroke="#ff9933"
      strokeWidth="4"
      strokeLinecap="round"
    />
    {/* White band with subtle shadow */}
    <path
      d="M2 17C32 11 62 7 102 13C132 18 147 12 162 8"
      stroke="#ffffff"
      strokeWidth="4"
      strokeLinecap="round"
      opacity="0.95"
    />
    {/* India Green band */}
    <path
      d="M4 22C34 16 64 12 104 18C134 23 149 17 164 13"
      stroke="#138808"
      strokeWidth="4"
      strokeLinecap="round"
    />
  </svg>
)

/** Rashtrapati Bhavan / Central Secretariat Dome architectural engraving */
export const RashtrapatiBhavanEngraving: React.FC<{ className?: string; opacity?: number }> = ({
  className = '',
  opacity = 0.22,
}) => (
  <svg
    viewBox="0 0 240 180"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    style={{ opacity }}
  >
    {/* Finial / Spire */}
    <line x1="120" y1="10" x2="120" y2="35" stroke="#7a5822" strokeWidth="2" />
    <circle cx="120" cy="8" r="3" fill="#7a5822" />

    {/* Classic Lutyens Dome */}
    <path
      d="M80 65C80 40 95 30 120 30C145 30 160 40 160 65H80Z"
      fill="#a47e3b"
      fillOpacity="0.15"
      stroke="#7a5822"
      strokeWidth="2"
    />
    {/* Dome ribs */}
    <path d="M100 65C100 45 110 32 120 30" stroke="#7a5822" strokeWidth="1.2" />
    <path d="M140 65C140 45 130 32 120 30" stroke="#7a5822" strokeWidth="1.2" />

    {/* Drum colonnade with arched windows */}
    <rect x="75" y="65" width="90" height="18" stroke="#7a5822" strokeWidth="1.8" fill="#e8d8b8" fillOpacity="0.3" />
    {Array.from({ length: 9 }).map((_, i) => (
      <path
        key={i}
        d={`M${82 + i * 9.5} 80V72C${82 + i * 9.5} 70 ${86 + i * 9.5} 70 ${86 + i * 9.5} 72V80`}
        stroke="#7a5822"
        strokeWidth="1.2"
      />
    ))}

    {/* Heavy cornice & balustrade */}
    <rect x="68" y="83" width="104" height="6" fill="#7a5822" stroke="#7a5822" strokeWidth="1" />

    {/* Main Facade Portico with Classical Columns */}
    <rect x="50" y="89" width="140" height="70" fill="#a47e3b" fillOpacity="0.08" stroke="#7a5822" strokeWidth="1.5" />
    {Array.from({ length: 11 }).map((_, i) => (
      <line
        key={i}
        x1={58 + i * 12.4}
        y1="92"
        x2={58 + i * 12.4}
        y2="155"
        stroke="#7a5822"
        strokeWidth="2"
      />
    ))}

    {/* Classical Pediment Gable on Left / Right wings */}
    <path d="M30 115L60 95L90 115H30Z" stroke="#7a5822" strokeWidth="1.5" fill="#a47e3b" fillOpacity="0.12" />
    <path d="M150 115L180 95L210 115H150Z" stroke="#7a5822" strokeWidth="1.5" fill="#a47e3b" fillOpacity="0.12" />

    {/* Plinth and monumental staircase */}
    <rect x="25" y="155" width="190" height="8" stroke="#7a5822" strokeWidth="1.8" fill="#7a5822" fillOpacity="0.2" />
    <rect x="15" y="163" width="210" height="8" stroke="#7a5822" strokeWidth="1.8" fill="#7a5822" fillOpacity="0.3" />
    <line x1="0" y1="175" x2="240" y2="175" stroke="#7a5822" strokeWidth="2" />
  </svg>
)

/** Compass Rose Navigation Icon */
export const CompassRose: React.FC<{ className?: string; size?: number }> = ({
  className = '',
  size = 40,
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 60 60"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
  >
    <circle cx="30" cy="30" r="26" stroke="#c9a227" strokeWidth="1.2" strokeOpacity="0.6" />
    <circle cx="30" cy="30" r="22" stroke="#c9a227" strokeWidth="0.8" strokeDasharray="2 3" strokeOpacity="0.4" />
    {/* North needle */}
    <polygon points="30,8 34,28 30,25 26,28" fill="#c0392b" />
    {/* South needle */}
    <polygon points="30,52 34,32 30,35 26,32" fill="#0d1b2a" opacity="0.6" />
    {/* East / West needles */}
    <polygon points="52,30 32,34 35,30 32,26" fill="#0d1b2a" opacity="0.4" />
    <polygon points="8,30 28,34 25,30 28,26" fill="#0d1b2a" opacity="0.4" />
    {/* North Label 'N' */}
    <text x="30" y="6" textAnchor="middle" fontSize="6.5" fontWeight="bold" fill="#0d1b2a" fontFamily="'Playfair Display', serif">
      N
    </text>
  </svg>
)
