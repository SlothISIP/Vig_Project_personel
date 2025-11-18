/**
 * Product Flow Visualization Component
 * Animates products moving through the production line
 */

import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { createProductMaterial } from '@/utils/three';

interface Product {
  id: string;
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  isDefect: boolean;
  progress: number; // 0 to 1 along the path
}

interface ProductFlowProps {
  path: THREE.Vector3[]; // Array of waypoints
  productCount: number;
  speed: number;
  defectRate: number;
}

export function ProductFlow({
  path,
  productCount = 10,
  speed = 0.5,
  defectRate = 0.05,
}: ProductFlowProps) {
  const productsRef = useRef<Product[]>([]);

  // Initialize products
  useMemo(() => {
    if (productsRef.current.length === 0 && path.length > 1) {
      const products: Product[] = [];
      for (let i = 0; i < productCount; i++) {
        const progress = i / productCount;
        const isDefect = Math.random() < defectRate;
        const position = getPositionOnPath(path, progress);
        const velocity = new THREE.Vector3(0, 0, 0);

        products.push({
          id: `product-${i}`,
          position: position.clone(),
          velocity,
          isDefect,
          progress,
        });
      }
      productsRef.current = products;
    }
  }, [path, productCount, defectRate]);

  // Animate products along the path
  useFrame((state, delta) => {
    if (path.length < 2) return;

    productsRef.current.forEach((product) => {
      // Update progress along path
      product.progress += (speed * delta) / getTotalPathLength(path);

      // Loop back to start if reached end
      if (product.progress >= 1) {
        product.progress = 0;
        product.isDefect = Math.random() < defectRate;
      }

      // Update position
      const newPosition = getPositionOnPath(path, product.progress);
      product.position.copy(newPosition);

      // Calculate velocity for orientation
      const nextProgress = product.progress + 0.01;
      const nextPosition = getPositionOnPath(path, Math.min(nextProgress, 1));
      product.velocity.subVectors(nextPosition, product.position).normalize();
    });
  });

  return (
    <group>
      {productsRef.current.map((product, index) => (
        <ProductBox
          key={product.id}
          position={product.position}
          isDefect={product.isDefect}
          rotation={calculateRotationFromVelocity(product.velocity)}
        />
      ))}

      {/* Path visualization (debug) */}
      {path.length > 1 && (
        <PathLine points={path} color={0x64748b} opacity={0.3} />
      )}
    </group>
  );
}

/**
 * Individual product box component
 */
interface ProductBoxProps {
  position: THREE.Vector3;
  isDefect: boolean;
  rotation: THREE.Euler;
}

function ProductBox({ position, isDefect, rotation }: ProductBoxProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    // Slight bobbing animation
    if (meshRef.current) {
      meshRef.current.position.y = position.y + Math.sin(state.clock.elapsedTime * 3) * 0.02;
    }
  });

  return (
    <mesh
      ref={meshRef}
      position={[position.x, position.y, position.z]}
      rotation={rotation}
      castShadow
    >
      <boxGeometry args={[0.3, 0.3, 0.3]} />
      <primitive object={createProductMaterial(isDefect)} attach="material" />

      {/* Defect indicator */}
      {isDefect && (
        <mesh position={[0, 0.2, 0]}>
          <sphereGeometry args={[0.08, 16, 16]} />
          <meshBasicMaterial
            color={0xef4444}
            emissive={0xef4444}
            emissiveIntensity={0.8}
          />
        </mesh>
      )}
    </mesh>
  );
}

/**
 * Path line visualization component
 */
interface PathLineProps {
  points: THREE.Vector3[];
  color: number;
  opacity: number;
}

function PathLine({ points, color, opacity }: PathLineProps) {
  const lineGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    return geometry;
  }, [points]);

  return (
    <line geometry={lineGeometry}>
      <lineBasicMaterial color={color} transparent opacity={opacity} linewidth={2} />
    </line>
  );
}

/**
 * Utility: Get position on path based on progress (0-1)
 */
function getPositionOnPath(path: THREE.Vector3[], progress: number): THREE.Vector3 {
  if (path.length === 0) return new THREE.Vector3();
  if (path.length === 1) return path[0].clone();

  const totalLength = getTotalPathLength(path);
  const targetDistance = progress * totalLength;

  let accumulatedDistance = 0;
  for (let i = 0; i < path.length - 1; i++) {
    const segmentLength = path[i].distanceTo(path[i + 1]);
    if (accumulatedDistance + segmentLength >= targetDistance) {
      const segmentProgress = (targetDistance - accumulatedDistance) / segmentLength;
      return new THREE.Vector3().lerpVectors(path[i], path[i + 1], segmentProgress);
    }
    accumulatedDistance += segmentLength;
  }

  return path[path.length - 1].clone();
}

/**
 * Utility: Calculate total path length
 */
function getTotalPathLength(path: THREE.Vector3[]): number {
  let length = 0;
  for (let i = 0; i < path.length - 1; i++) {
    length += path[i].distanceTo(path[i + 1]);
  }
  return length;
}

/**
 * Utility: Calculate rotation from velocity vector
 */
function calculateRotationFromVelocity(velocity: THREE.Vector3): THREE.Euler {
  if (velocity.length() === 0) return new THREE.Euler(0, 0, 0);

  const angle = Math.atan2(velocity.x, velocity.z);
  return new THREE.Euler(0, angle, 0);
}
