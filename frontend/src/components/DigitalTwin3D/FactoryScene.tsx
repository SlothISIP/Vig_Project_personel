/**
 * Factory Scene Component
 * Main 3D scene with camera controls, lighting, and factory visualization
 */

import React, { Suspense, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, Sky } from '@react-three/drei';
import { FactoryLayout } from './FactoryLayout';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import type { MachineState } from '@/types';

interface FactorySceneProps {
  machines: MachineState[];
  onMachineClick?: (machineId: string) => void;
  showProductFlow?: boolean;
  cameraPosition?: [number, number, number];
}

export function FactoryScene({
  machines,
  onMachineClick,
  showProductFlow = true,
  cameraPosition = [15, 12, 15],
}: FactorySceneProps) {
  const [enableShadows, setEnableShadows] = useState(true);

  return (
    <div className="w-full h-full relative">
      {/* 3D Canvas */}
      <Canvas
        shadows={enableShadows}
        gl={{ antialias: true, alpha: false }}
        dpr={[1, 2]}
        className="bg-gradient-to-b from-sky-100 to-sky-200"
      >
        <Suspense fallback={null}>
          {/* Camera */}
          <PerspectiveCamera
            makeDefault
            position={cameraPosition}
            fov={50}
            near={0.1}
            far={1000}
          />

          {/* Camera controls */}
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={5}
            maxDistance={50}
            maxPolarAngle={Math.PI / 2.1}
            target={[0, 0, 0]}
          />

          {/* Lighting */}
          <SceneLighting />

          {/* Environment */}
          <Environment preset="warehouse" />

          {/* Sky */}
          <Sky
            distance={450000}
            sunPosition={[10, 8, 10]}
            inclination={0.6}
            azimuth={0.25}
          />

          {/* Factory layout and machines */}
          <FactoryLayout
            machines={machines}
            onMachineClick={onMachineClick}
            showProductFlow={showProductFlow}
          />

          {/* Fog for depth */}
          <fog attach="fog" args={['#e0f2fe', 30, 100]} />
        </Suspense>
      </Canvas>

      {/* Controls overlay */}
      <SceneControls
        enableShadows={enableShadows}
        onToggleShadows={() => setEnableShadows(!enableShadows)}
      />
    </div>
  );
}

/**
 * Scene lighting setup
 */
function SceneLighting() {
  return (
    <>
      {/* Ambient light for base illumination */}
      <ambientLight intensity={0.4} color="#ffffff" />

      {/* Main directional light (sun) */}
      <directionalLight
        position={[10, 15, 10]}
        intensity={1.2}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-far={50}
        shadow-camera-left={-25}
        shadow-camera-right={25}
        shadow-camera-top={25}
        shadow-camera-bottom={-25}
        shadow-bias={-0.0001}
      />

      {/* Fill light from the side */}
      <directionalLight
        position={[-10, 8, -5]}
        intensity={0.4}
        color="#e0f2fe"
      />

      {/* Hemisphere light for natural ambient */}
      <hemisphereLight
        args={['#87ceeb', '#f1f5f9', 0.3]}
        position={[0, 20, 0]}
      />
    </>
  );
}

/**
 * Scene controls overlay
 */
interface SceneControlsProps {
  enableShadows: boolean;
  onToggleShadows: () => void;
}

function SceneControls({ enableShadows, onToggleShadows }: SceneControlsProps) {
  return (
    <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm rounded-lg shadow-lg p-4 space-y-3">
      <div className="text-sm font-semibold text-gray-700 mb-2">
        3D Controls
      </div>

      {/* Camera controls info */}
      <div className="text-xs text-gray-600 space-y-1">
        <div className="flex items-center gap-2">
          <div className="w-20 font-medium">Rotate:</div>
          <div>Left Click + Drag</div>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-20 font-medium">Pan:</div>
          <div>Right Click + Drag</div>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-20 font-medium">Zoom:</div>
          <div>Scroll Wheel</div>
        </div>
      </div>

      <div className="border-t border-gray-200 pt-3">
        {/* Toggle shadows */}
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={enableShadows}
            onChange={onToggleShadows}
            className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
          />
          <span className="text-sm text-gray-700">Enable Shadows</span>
        </label>
      </div>

      {/* Legend */}
      <div className="border-t border-gray-200 pt-3">
        <div className="text-xs font-semibold text-gray-700 mb-2">Status Colors</div>
        <div className="space-y-1 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span>Running</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <span>Warning</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <span>Error</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gray-400"></div>
            <span>Idle</span>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Loading fallback component
 */
export function FactorySceneLoader() {
  return (
    <div className="w-full h-full flex items-center justify-center bg-gradient-to-b from-sky-100 to-sky-200">
      <div className="text-center">
        <LoadingSpinner size="large" />
        <p className="mt-4 text-gray-600 font-medium">Loading 3D Factory Scene...</p>
      </div>
    </div>
  );
}
