# Digital Twin 3D Visualization

Real-time 3D factory visualization using React Three Fiber and Three.js.

## Features

- **Interactive 3D Scene**: Full camera controls (rotate, pan, zoom)
- **Real-time Updates**: Machine status and health visualized in 3D
- **Product Flow Animation**: Animated products moving through production lines
- **Status Indicators**: Color-coded machines based on status and health
- **Sensor Visualization**: IoT sensors displayed with status indicators
- **Health Bars**: Visual health indicators above each machine
- **Responsive Lighting**: Dynamic shadows and realistic lighting
- **Performance Optimized**: Efficient rendering with React Three Fiber

## Components

### FactoryScene

Main 3D scene component with camera, lighting, and controls.

```tsx
import { FactoryScene } from '@/components/DigitalTwin3D';

<FactoryScene
  machines={machineData}
  onMachineClick={(machineId) => console.log('Clicked:', machineId)}
  showProductFlow={true}
  cameraPosition={[15, 12, 15]}
/>
```

**Props:**
- `machines`: Array of MachineState objects
- `onMachineClick`: Callback when a machine is clicked
- `showProductFlow`: Enable/disable product animation
- `cameraPosition`: Initial camera position [x, y, z]

### Machine3D

Individual machine component with status visualization.

```tsx
<Machine3D
  machineId="M001"
  machineName="Assembly Station 1"
  position={[0, 1.25, 0]}
  status="running"
  healthScore={0.85}
  temperature={72.5}
  vibration={2.3}
  defectRate={0.05}
  onClick={handleClick}
/>
```

### ProductFlow

Animated product flow along a defined path.

```tsx
<ProductFlow
  path={waypoints}
  productCount={10}
  speed={1.5}
  defectRate={0.05}
/>
```

### FactoryLayout

Complete factory floor with machines, conveyors, and infrastructure.

```tsx
<FactoryLayout
  machines={machines}
  onMachineClick={handleClick}
  showProductFlow={true}
/>
```

## Status Colors

Machines are color-coded based on their status:

- **Green**: Running normally
- **Yellow/Amber**: Warning state
- **Red**: Error state
- **Gray**: Idle/offline
- **Blue**: Maintenance mode

Health scores blend with status colors to show overall machine condition.

## Camera Controls

- **Rotate**: Left click + drag
- **Pan**: Right click + drag
- **Zoom**: Scroll wheel
- **Reset**: Double-click

## Utilities

### Colors (`utils/three/colors.ts`)

```ts
import { STATUS_COLORS, getHealthColor, URGENCY_COLORS } from '@/utils/three/colors';

const statusColor = STATUS_COLORS.running;
const healthColor = getHealthColor(0.75);
```

### Geometries (`utils/three/geometries.ts`)

```ts
import { createMachineGeometry, createProductGeometry } from '@/utils/three/geometries';

const machineGroup = createMachineGeometry(2, 2.5, 1.5);
const productMesh = createProductGeometry(0.3);
```

### Materials (`utils/three/materials.ts`)

```ts
import { createMachineMaterial, updateMachineMaterial } from '@/utils/three/materials';

const material = createMachineMaterial('running', 0.85);
updateMachineMaterial(material, 'warning', 0.70);
```

## Performance Considerations

1. **Shadow Optimization**: Shadows can be disabled in the UI for better performance
2. **Level of Detail**: Simpler geometries used for distant objects
3. **Instancing**: Repeating elements use instanced rendering
4. **Frame Rate**: Target 60 FPS with fallback to 30 FPS on lower-end devices

## Customization

### Factory Layout

Modify `FactoryLayout.tsx` to adjust:
- Machine positions
- Conveyor belt paths
- Factory dimensions
- Wall configuration

### Visual Style

Adjust materials and colors in:
- `utils/three/colors.ts` - Color schemes
- `utils/three/materials.ts` - Material properties
- `FactoryScene.tsx` - Lighting setup

### Animation

Control animation in:
- `ProductFlow.tsx` - Product movement speed and behavior
- `Machine3D.tsx` - Status indicator pulsing and effects

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

WebGL 2.0 required for full features.

## Dependencies

- `three`: ^0.160.0
- `@react-three/fiber`: ^8.15.0
- `@react-three/drei`: ^9.93.0
- `@types/three`: ^0.160.0

## Examples

### Basic Usage

```tsx
import { FactoryScene } from '@/components/DigitalTwin3D';
import { apiClient } from '@/services/api';

function MyComponent() {
  const [machines, setMachines] = useState([]);

  useEffect(() => {
    apiClient.getFactoryState('factory-1')
      .then(data => setMachines(data.machines));
  }, []);

  return (
    <div style={{ width: '100%', height: '600px' }}>
      <FactoryScene machines={machines} />
    </div>
  );
}
```

### Split View

```tsx
<div className="flex">
  <div className="w-2/3">
    <FactoryScene machines={machines} />
  </div>
  <div className="w-1/3">
    {/* 2D controls and details */}
  </div>
</div>
```

## Troubleshooting

### Poor Performance

1. Disable shadows in scene controls
2. Reduce `productCount` in ProductFlow
3. Lower `dpr` (device pixel ratio) in Canvas

### Blank Scene

1. Check machine data is loading correctly
2. Verify camera position is valid
3. Check browser console for WebGL errors

### Click Detection Not Working

1. Ensure `onMachineClick` prop is provided
2. Check machine positions don't overlap
3. Verify pointer events aren't blocked by CSS
