/**
 * Color schemes for 3D visualization
 * Maps status and health to Three.js compatible colors
 */

import * as THREE from 'three';

/**
 * Status color mapping
 */
export const STATUS_COLORS = {
  idle: new THREE.Color(0x9ca3af), // gray
  running: new THREE.Color(0x10b981), // green
  warning: new THREE.Color(0xf59e0b), // yellow/amber
  error: new THREE.Color(0xef4444), // red
  maintenance: new THREE.Color(0x3b82f6), // blue
  offline: new THREE.Color(0x6b7280), // dark gray
} as const;

/**
 * Health score color gradient
 * Returns color based on health percentage (0-1)
 */
export function getHealthColor(healthScore: number): THREE.Color {
  if (healthScore >= 0.8) {
    return new THREE.Color(0x10b981); // green
  } else if (healthScore >= 0.6) {
    return new THREE.Color(0x84cc16); // lime
  } else if (healthScore >= 0.4) {
    return new THREE.Color(0xf59e0b); // amber
  } else if (healthScore >= 0.2) {
    return new THREE.Color(0xf97316); // orange
  } else {
    return new THREE.Color(0xef4444); // red
  }
}

/**
 * Urgency color mapping for predictive maintenance
 */
export const URGENCY_COLORS = {
  none: new THREE.Color(0x10b981), // green
  low: new THREE.Color(0x84cc16), // lime
  medium: new THREE.Color(0xf59e0b), // amber
  high: new THREE.Color(0xf97316), // orange
  critical: new THREE.Color(0xef4444), // red
} as const;

/**
 * Material type colors
 */
export const MATERIAL_COLORS = {
  machine: new THREE.Color(0x64748b), // slate
  conveyor: new THREE.Color(0x475569), // dark slate
  product: new THREE.Color(0x3b82f6), // blue
  defect: new THREE.Color(0xef4444), // red
  floor: new THREE.Color(0xf1f5f9), // light gray
  wall: new THREE.Color(0xe2e8f0), // very light gray
} as const;

/**
 * Get interpolated color between two colors
 */
export function lerpColor(
  color1: THREE.Color,
  color2: THREE.Color,
  t: number
): THREE.Color {
  const result = new THREE.Color();
  result.r = color1.r + (color2.r - color1.r) * t;
  result.g = color1.g + (color2.g - color1.g) * t;
  result.b = color1.b + (color2.b - color1.b) * t;
  return result;
}

/**
 * Convert hex color string to THREE.Color
 */
export function hexToColor(hex: string): THREE.Color {
  return new THREE.Color(hex);
}

/**
 * Get emissive color for glowing effects based on status
 */
export function getEmissiveColor(status: string, intensity: number = 0.3): THREE.Color {
  const baseColor = STATUS_COLORS[status as keyof typeof STATUS_COLORS] || STATUS_COLORS.idle;
  return new THREE.Color(
    baseColor.r * intensity,
    baseColor.g * intensity,
    baseColor.b * intensity
  );
}
