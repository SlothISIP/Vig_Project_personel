/**
 * Reusable 3D geometries for factory components
 */

import * as THREE from 'three';

/**
 * Create a machine geometry (box-like structure with details)
 */
export function createMachineGeometry(
  width: number = 2,
  height: number = 2.5,
  depth: number = 1.5
): THREE.Group {
  const group = new THREE.Group();

  // Main body
  const bodyGeometry = new THREE.BoxGeometry(width, height, depth);
  const body = new THREE.Mesh(bodyGeometry);
  group.add(body);

  // Base platform
  const baseGeometry = new THREE.BoxGeometry(width * 1.2, 0.2, depth * 1.2);
  const base = new THREE.Mesh(baseGeometry);
  base.position.y = -height / 2 - 0.1;
  group.add(base);

  // Control panel
  const panelGeometry = new THREE.BoxGeometry(width * 0.3, height * 0.4, 0.1);
  const panel = new THREE.Mesh(panelGeometry);
  panel.position.set(width * 0.35, height * 0.2, depth / 2 + 0.05);
  group.add(panel);

  return group;
}

/**
 * Create a conveyor belt geometry
 */
export function createConveyorGeometry(
  length: number = 10,
  width: number = 0.8,
  height: number = 0.3
): THREE.Group {
  const group = new THREE.Group();

  // Belt surface
  const beltGeometry = new THREE.BoxGeometry(length, height * 0.3, width);
  const belt = new THREE.Mesh(beltGeometry);
  belt.position.y = height / 2;
  group.add(belt);

  // Support frame
  const frameGeometry = new THREE.BoxGeometry(length, height, 0.1);
  const frameSide1 = new THREE.Mesh(frameGeometry);
  frameSide1.position.set(0, 0, width / 2);
  group.add(frameSide1);

  const frameSide2 = new THREE.Mesh(frameGeometry);
  frameSide2.position.set(0, 0, -width / 2);
  group.add(frameSide2);

  // Rollers
  const rollerRadius = 0.15;
  const rollerGeometry = new THREE.CylinderGeometry(rollerRadius, rollerRadius, width, 16);
  const rollerCount = Math.floor(length / 2);

  for (let i = 0; i < rollerCount; i++) {
    const roller = new THREE.Mesh(rollerGeometry);
    roller.rotation.z = Math.PI / 2;
    roller.position.set(-length / 2 + (i * length) / rollerCount, height / 2, 0);
    group.add(roller);
  }

  return group;
}

/**
 * Create a product box geometry
 */
export function createProductGeometry(size: number = 0.3): THREE.Mesh {
  const geometry = new THREE.BoxGeometry(size, size, size);
  const mesh = new THREE.Mesh(geometry);
  return mesh;
}

/**
 * Create a defect indicator geometry (warning symbol)
 */
export function createDefectIndicator(): THREE.Group {
  const group = new THREE.Group();

  // Warning triangle
  const shape = new THREE.Shape();
  shape.moveTo(0, 0.5);
  shape.lineTo(-0.4, -0.3);
  shape.lineTo(0.4, -0.3);
  shape.lineTo(0, 0.5);

  const extrudeSettings = {
    depth: 0.1,
    bevelEnabled: false,
  };

  const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
  const mesh = new THREE.Mesh(geometry);
  group.add(mesh);

  // Exclamation mark
  const dotGeometry = new THREE.SphereGeometry(0.05, 16, 16);
  const dot = new THREE.Mesh(dotGeometry);
  dot.position.set(0, -0.15, 0.1);
  group.add(dot);

  const lineGeometry = new THREE.BoxGeometry(0.08, 0.25, 0.08);
  const line = new THREE.Mesh(lineGeometry);
  line.position.set(0, 0.05, 0.1);
  group.add(line);

  group.scale.set(0.5, 0.5, 0.5);
  return group;
}

/**
 * Create factory floor grid
 */
export function createFloorGrid(
  size: number = 50,
  divisions: number = 50
): THREE.GridHelper {
  const gridHelper = new THREE.GridHelper(
    size,
    divisions,
    0xcccccc,
    0xe5e5e5
  );
  return gridHelper;
}

/**
 * Create a sensor indicator (small sphere with glow)
 */
export function createSensorGeometry(radius: number = 0.1): THREE.Mesh {
  const geometry = new THREE.SphereGeometry(radius, 16, 16);
  const mesh = new THREE.Mesh(geometry);
  return mesh;
}

/**
 * Create an arrow helper for flow direction
 */
export function createFlowArrow(
  direction: THREE.Vector3,
  origin: THREE.Vector3,
  length: number = 1,
  color: number = 0x3b82f6
): THREE.ArrowHelper {
  return new THREE.ArrowHelper(
    direction.normalize(),
    origin,
    length,
    color,
    length * 0.2,
    length * 0.15
  );
}

/**
 * Create a health indicator bar above machines
 */
export function createHealthBar(
  width: number = 1,
  height: number = 0.1
): THREE.Group {
  const group = new THREE.Group();

  // Background
  const bgGeometry = new THREE.PlaneGeometry(width, height);
  const bgMaterial = new THREE.MeshBasicMaterial({ color: 0x333333, transparent: true, opacity: 0.5 });
  const background = new THREE.Mesh(bgGeometry, bgMaterial);
  group.add(background);

  // Foreground bar (will be scaled based on health)
  const fgGeometry = new THREE.PlaneGeometry(width, height);
  const foreground = new THREE.Mesh(fgGeometry);
  foreground.name = 'healthBar';
  foreground.position.z = 0.01;
  group.add(foreground);

  return group;
}

/**
 * Create a text sprite for labels
 */
export function createTextSprite(
  text: string,
  fontSize: number = 64,
  fontColor: string = '#ffffff',
  backgroundColor: string = 'rgba(0,0,0,0.7)'
): THREE.Sprite {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d')!;
  canvas.width = 512;
  canvas.height = 128;

  // Background
  context.fillStyle = backgroundColor;
  context.fillRect(0, 0, canvas.width, canvas.height);

  // Text
  context.font = `${fontSize}px Arial`;
  context.fillStyle = fontColor;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(text, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(2, 0.5, 1);

  return sprite;
}
