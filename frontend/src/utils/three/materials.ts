/**
 * Material definitions for 3D visualization
 */

import * as THREE from 'three';
import { STATUS_COLORS, MATERIAL_COLORS, getHealthColor } from './colors';

/**
 * Create standard machine material with status-based color
 */
export function createMachineMaterial(
  status: string = 'idle',
  healthScore: number = 1.0
): THREE.MeshStandardMaterial {
  const statusColor = STATUS_COLORS[status as keyof typeof STATUS_COLORS] || STATUS_COLORS.idle;
  const healthColor = getHealthColor(healthScore);

  // Blend status and health colors
  const blendedColor = new THREE.Color(
    (statusColor.r + healthColor.r) / 2,
    (statusColor.g + healthColor.g) / 2,
    (statusColor.b + healthColor.b) / 2
  );

  return new THREE.MeshStandardMaterial({
    color: blendedColor,
    metalness: 0.6,
    roughness: 0.4,
    emissive: statusColor,
    emissiveIntensity: status === 'running' ? 0.2 : 0.05,
  });
}

/**
 * Create conveyor belt material with animated texture
 */
export function createConveyorMaterial(): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: MATERIAL_COLORS.conveyor,
    metalness: 0.3,
    roughness: 0.7,
  });
}

/**
 * Create product material
 */
export function createProductMaterial(isDefect: boolean = false): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: isDefect ? MATERIAL_COLORS.defect : MATERIAL_COLORS.product,
    metalness: 0.4,
    roughness: 0.5,
    emissive: isDefect ? MATERIAL_COLORS.defect : new THREE.Color(0x000000),
    emissiveIntensity: isDefect ? 0.3 : 0,
  });
}

/**
 * Create floor material
 */
export function createFloorMaterial(): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: MATERIAL_COLORS.floor,
    metalness: 0.1,
    roughness: 0.9,
  });
}

/**
 * Create wall material
 */
export function createWallMaterial(): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: MATERIAL_COLORS.wall,
    metalness: 0,
    roughness: 1,
  });
}

/**
 * Create sensor indicator material with pulsing emissive
 */
export function createSensorMaterial(
  isActive: boolean = true,
  sensorType: string = 'temperature'
): THREE.MeshStandardMaterial {
  const colorMap: Record<string, number> = {
    temperature: 0xef4444, // red
    vibration: 0xf59e0b, // amber
    pressure: 0x3b82f6, // blue
    speed: 0x10b981, // green
  };

  const color = colorMap[sensorType] || 0x9ca3af;

  return new THREE.MeshStandardMaterial({
    color: color,
    metalness: 0.8,
    roughness: 0.2,
    emissive: new THREE.Color(color),
    emissiveIntensity: isActive ? 0.5 : 0.1,
  });
}

/**
 * Create transparent health bar material
 */
export function createHealthBarMaterial(healthScore: number): THREE.MeshBasicMaterial {
  const color = getHealthColor(healthScore);
  return new THREE.MeshBasicMaterial({
    color: color,
    transparent: true,
    opacity: 0.8,
  });
}

/**
 * Create glow material for highlighting
 */
export function createGlowMaterial(color: THREE.Color, intensity: number = 1): THREE.MeshBasicMaterial {
  return new THREE.MeshBasicMaterial({
    color: color,
    transparent: true,
    opacity: 0.3 * intensity,
    blending: THREE.AdditiveBlending,
  });
}

/**
 * Create wireframe material for debug mode
 */
export function createWireframeMaterial(color: number = 0x00ff00): THREE.MeshBasicMaterial {
  return new THREE.MeshBasicMaterial({
    color: color,
    wireframe: true,
    transparent: true,
    opacity: 0.3,
  });
}

/**
 * Update material based on status change
 */
export function updateMachineMaterial(
  material: THREE.MeshStandardMaterial,
  status: string,
  healthScore: number
): void {
  const statusColor = STATUS_COLORS[status as keyof typeof STATUS_COLORS] || STATUS_COLORS.idle;
  const healthColor = getHealthColor(healthScore);

  // Blend colors
  const blendedColor = new THREE.Color(
    (statusColor.r + healthColor.r) / 2,
    (statusColor.g + healthColor.g) / 2,
    (statusColor.b + healthColor.b) / 2
  );

  material.color.copy(blendedColor);
  material.emissive.copy(statusColor);
  material.emissiveIntensity = status === 'running' ? 0.2 : 0.05;
  material.needsUpdate = true;
}

/**
 * Create particle material for visual effects
 */
export function createParticleMaterial(
  color: number = 0x3b82f6,
  size: number = 0.1
): THREE.PointsMaterial {
  return new THREE.PointsMaterial({
    color: color,
    size: size,
    transparent: true,
    opacity: 0.6,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
}
