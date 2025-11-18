/**
 * Factory Layout Component
 * Manages the overall factory floor layout with machines, conveyors, and infrastructure
 */

import React from 'react';
import * as THREE from 'three';
import { Machine3D } from './Machine3D';
import { ProductFlow } from './ProductFlow';
import { createConveyorMaterial, createFloorMaterial } from '@/utils/three';
import type { MachineState } from '@/types';

interface FactoryLayoutProps {
  machines: MachineState[];
  onMachineClick?: (machineId: string) => void;
  showProductFlow?: boolean;
}

export function FactoryLayout({
  machines,
  onMachineClick,
  showProductFlow = true,
}: FactoryLayoutProps) {
  // Define production line layout (3 machines in a line)
  const machinePositions: Record<string, [number, number, number]> = {
    'M001': [-8, 1.25, 0],
    'M002': [0, 1.25, 0],
    'M003': [8, 1.25, 0],
  };

  // Define conveyor belt positions
  const conveyorSegments = [
    { start: [-8, 0.15, 2], end: [0, 0.15, 2], length: 8 },
    { start: [0, 0.15, 2], end: [8, 0.15, 2], length: 8 },
  ];

  // Product flow path (waypoints)
  const productPath = [
    new THREE.Vector3(-10, 0.5, 2),
    new THREE.Vector3(-8, 0.5, 2),
    new THREE.Vector3(-4, 0.5, 2),
    new THREE.Vector3(0, 0.5, 2),
    new THREE.Vector3(4, 0.5, 2),
    new THREE.Vector3(8, 0.5, 2),
    new THREE.Vector3(10, 0.5, 2),
  ];

  return (
    <group>
      {/* Floor */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[50, 50]} />
        <primitive object={createFloorMaterial()} attach="material" />
      </mesh>

      {/* Floor grid */}
      <gridHelper args={[50, 50, 0xcccccc, 0xe5e5e5]} position={[0, 0.01, 0]} />

      {/* Factory walls */}
      <FactoryWalls />

      {/* Conveyor belts */}
      {conveyorSegments.map((segment, index) => (
        <ConveyorBelt
          key={index}
          start={segment.start}
          end={segment.end}
          length={segment.length}
        />
      ))}

      {/* Machines */}
      {machines.map((machine) => {
        const position = machinePositions[machine.machine_id] || [0, 1.25, 0];
        return (
          <Machine3D
            key={machine.machine_id}
            machineId={machine.machine_id}
            machineName={machine.machine_name}
            position={position as [number, number, number]}
            status={machine.status}
            healthScore={machine.health_score}
            temperature={machine.temperature}
            vibration={machine.vibration}
            defectRate={machine.defect_rate}
            onClick={onMachineClick}
          />
        );
      })}

      {/* Product flow animation */}
      {showProductFlow && (
        <ProductFlow
          path={productPath}
          productCount={12}
          speed={1.5}
          defectRate={0.08}
        />
      )}

      {/* Lighting fixtures */}
      <FactoryLights />

      {/* Support structures */}
      <SupportPillars />
    </group>
  );
}

/**
 * Conveyor belt segment component
 */
interface ConveyorBeltProps {
  start: number[];
  end: number[];
  length: number;
}

function ConveyorBelt({ start, end, length }: ConveyorBeltProps) {
  const width = 0.8;
  const height = 0.3;

  const position = [
    (start[0] + end[0]) / 2,
    start[1],
    (start[2] + end[2]) / 2,
  ] as [number, number, number];

  const angle = Math.atan2(end[2] - start[2], end[0] - start[0]);

  return (
    <group position={position} rotation={[0, angle, 0]}>
      {/* Belt surface */}
      <mesh position={[0, height / 2, 0]} castShadow receiveShadow>
        <boxGeometry args={[length, height * 0.3, width]} />
        <primitive object={createConveyorMaterial()} attach="material" />
      </mesh>

      {/* Support frame sides */}
      <mesh position={[0, 0, width / 2]} receiveShadow>
        <boxGeometry args={[length, height, 0.05]} />
        <meshStandardMaterial color={0x475569} metalness={0.6} roughness={0.4} />
      </mesh>

      <mesh position={[0, 0, -width / 2]} receiveShadow>
        <boxGeometry args={[length, height, 0.05]} />
        <meshStandardMaterial color={0x475569} metalness={0.6} roughness={0.4} />
      </mesh>

      {/* Rollers */}
      {Array.from({ length: Math.floor(length / 0.5) }).map((_, i) => {
        const x = -length / 2 + (i * length) / Math.floor(length / 0.5);
        return (
          <mesh
            key={i}
            position={[x, height / 2, 0]}
            rotation={[0, 0, Math.PI / 2]}
          >
            <cylinderGeometry args={[0.08, 0.08, width, 12]} />
            <meshStandardMaterial color={0x1e293b} metalness={0.8} roughness={0.3} />
          </mesh>
        );
      })}
    </group>
  );
}

/**
 * Factory walls component
 */
function FactoryWalls() {
  const wallHeight = 5;
  const wallThickness = 0.3;
  const factorySize = 25;

  return (
    <group>
      {/* Back wall */}
      <mesh position={[0, wallHeight / 2, -factorySize / 2]} receiveShadow>
        <boxGeometry args={[factorySize, wallHeight, wallThickness]} />
        <meshStandardMaterial color={0xe2e8f0} roughness={0.9} />
      </mesh>

      {/* Side walls */}
      <mesh position={[-factorySize / 2, wallHeight / 2, 0]} receiveShadow>
        <boxGeometry args={[wallThickness, wallHeight, factorySize]} />
        <meshStandardMaterial color={0xe2e8f0} roughness={0.9} />
      </mesh>

      <mesh position={[factorySize / 2, wallHeight / 2, 0]} receiveShadow>
        <boxGeometry args={[wallThickness, wallHeight, factorySize]} />
        <meshStandardMaterial color={0xe2e8f0} roughness={0.9} />
      </mesh>
    </group>
  );
}

/**
 * Factory overhead lights
 */
function FactoryLights() {
  const lightPositions = [
    [-8, 4.5, 0],
    [0, 4.5, 0],
    [8, 4.5, 0],
  ];

  return (
    <group>
      {lightPositions.map((pos, i) => (
        <group key={i} position={pos as [number, number, number]}>
          {/* Light fixture */}
          <mesh>
            <cylinderGeometry args={[0.4, 0.5, 0.2, 16]} />
            <meshStandardMaterial color={0x334155} metalness={0.8} roughness={0.3} />
          </mesh>

          {/* Light glow */}
          <mesh position={[0, -0.15, 0]}>
            <circleGeometry args={[0.5, 32]} />
            <meshBasicMaterial
              color={0xfef3c7}
              transparent
              opacity={0.6}
            />
          </mesh>

          {/* Point light */}
          <pointLight
            color={0xffffff}
            intensity={0.5}
            distance={15}
            decay={2}
            castShadow
          />
        </group>
      ))}
    </group>
  );
}

/**
 * Support pillars
 */
function SupportPillars() {
  const pillarPositions = [
    [-12, 2.5, -8],
    [-12, 2.5, 8],
    [12, 2.5, -8],
    [12, 2.5, 8],
  ];

  return (
    <group>
      {pillarPositions.map((pos, i) => (
        <mesh key={i} position={pos as [number, number, number]} receiveShadow castShadow>
          <cylinderGeometry args={[0.3, 0.3, 5, 16]} />
          <meshStandardMaterial color={0x64748b} metalness={0.5} roughness={0.6} />
        </mesh>
      ))}
    </group>
  );
}
