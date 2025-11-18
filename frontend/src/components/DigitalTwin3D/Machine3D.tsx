/**
 * 3D Machine Component
 * Renders individual machine with status indicators and health visualization
 */

import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';
import {
  createMachineMaterial,
  createHealthBarMaterial,
  createSensorMaterial,
  updateMachineMaterial,
  getHealthColor,
} from '@/utils/three';

interface Machine3DProps {
  machineId: string;
  machineName: string;
  position: [number, number, number];
  status: string;
  healthScore: number;
  temperature?: number;
  vibration?: number;
  defectRate?: number;
  onClick?: (machineId: string) => void;
}

export function Machine3D({
  machineId,
  machineName,
  position,
  status,
  healthScore,
  temperature,
  vibration,
  defectRate,
  onClick,
}: Machine3DProps) {
  const groupRef = useRef<THREE.Group>(null);
  const healthBarRef = useRef<THREE.Mesh>(null);
  const machineMaterialRef = useRef<THREE.MeshStandardMaterial>(
    createMachineMaterial(status, healthScore)
  );

  // Dimensions
  const width = 2;
  const height = 2.5;
  const depth = 1.5;

  // Update material when status or health changes
  useEffect(() => {
    if (machineMaterialRef.current) {
      updateMachineMaterial(machineMaterialRef.current, status, healthScore);
    }
  }, [status, healthScore]);

  // Pulsing animation for running machines
  useFrame((state) => {
    if (groupRef.current && status === 'running') {
      const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.05 + 1;
      machineMaterialRef.current.emissiveIntensity = 0.2 * pulse;
    }

    // Update health bar scale
    if (healthBarRef.current) {
      healthBarRef.current.scale.x = healthScore;
      const healthMaterial = healthBarRef.current.material as THREE.MeshBasicMaterial;
      healthMaterial.color = getHealthColor(healthScore);
    }
  });

  // Sensor positions
  const sensorPositions = useMemo(() => {
    return [
      { pos: [width / 2 + 0.2, height / 4, 0] as [number, number, number], type: 'temperature' },
      { pos: [-width / 2 - 0.2, height / 4, 0] as [number, number, number], type: 'vibration' },
      { pos: [0, height / 2 + 0.2, depth / 2] as [number, number, number], type: 'pressure' },
    ];
  }, [width, height, depth]);

  const handleClick = () => {
    if (onClick) {
      onClick(machineId);
    }
  };

  return (
    <group ref={groupRef} position={position} onClick={handleClick}>
      {/* Main machine body */}
      <mesh material={machineMaterialRef.current} castShadow receiveShadow>
        <boxGeometry args={[width, height, depth]} />
      </mesh>

      {/* Base platform */}
      <mesh
        position={[0, -height / 2 - 0.1, 0]}
        material={machineMaterialRef.current}
        castShadow
        receiveShadow
      >
        <boxGeometry args={[width * 1.2, 0.2, depth * 1.2]} />
      </mesh>

      {/* Control panel */}
      <mesh
        position={[width * 0.35, height * 0.2, depth / 2 + 0.05]}
        castShadow
        receiveShadow
      >
        <boxGeometry args={[width * 0.3, height * 0.4, 0.1]} />
        <meshStandardMaterial color={0x1e293b} metalness={0.8} roughness={0.3} />
      </mesh>

      {/* Display screen on control panel */}
      <mesh
        position={[width * 0.35, height * 0.2, depth / 2 + 0.11]}
      >
        <planeGeometry args={[width * 0.25, height * 0.25]} />
        <meshBasicMaterial
          color={status === 'running' ? 0x10b981 : status === 'error' ? 0xef4444 : 0x334155}
          emissive={status === 'running' ? 0x10b981 : status === 'error' ? 0xef4444 : 0x000000}
          emissiveIntensity={0.5}
        />
      </mesh>

      {/* Sensors */}
      {sensorPositions.map((sensor, index) => (
        <mesh
          key={index}
          position={sensor.pos}
          material={createSensorMaterial(status === 'running', sensor.type)}
        >
          <sphereGeometry args={[0.1, 16, 16]} />
        </mesh>
      ))}

      {/* Health bar above machine */}
      <group position={[0, height / 2 + 0.5, 0]}>
        {/* Background */}
        <mesh>
          <planeGeometry args={[width, 0.15]} />
          <meshBasicMaterial color={0x333333} transparent opacity={0.5} />
        </mesh>

        {/* Foreground health indicator */}
        <mesh
          ref={healthBarRef}
          position={[-(width / 2) * (1 - healthScore), 0, 0.01]}
        >
          <planeGeometry args={[width, 0.15]} />
          <primitive object={createHealthBarMaterial(healthScore)} attach="material" />
        </mesh>
      </group>

      {/* Machine label */}
      <Text
        position={[0, height / 2 + 0.8, 0]}
        fontSize={0.3}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.02}
        outlineColor="#000000"
      >
        {machineName}
      </Text>

      {/* Status label */}
      <Text
        position={[0, height / 2 + 1.1, 0]}
        fontSize={0.2}
        color={
          status === 'running'
            ? '#10b981'
            : status === 'error'
            ? '#ef4444'
            : status === 'warning'
            ? '#f59e0b'
            : '#9ca3af'
        }
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.01}
        outlineColor="#000000"
      >
        {status.toUpperCase()}
      </Text>

      {/* Defect rate indicator (if high) */}
      {defectRate !== undefined && defectRate > 0.05 && (
        <mesh position={[0, height / 2 + 0.3, depth / 2 + 0.1]}>
          <sphereGeometry args={[0.15, 16, 16]} />
          <meshBasicMaterial
            color={0xef4444}
            emissive={0xef4444}
            emissiveIntensity={0.8}
          />
        </mesh>
      )}
    </group>
  );
}
