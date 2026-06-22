import * as THREE from "three";

const viewport = document.querySelector("#viewport");
const playButton = document.querySelector("#playButton");
const timeline = document.querySelector("#timeline");
const speed = document.querySelector("#speed");
const resultBadge = document.querySelector("#resultBadge");
const stepLabel = document.querySelector("#stepLabel");
const poseLabel = document.querySelector("#poseLabel");
const thoughtText = document.querySelector("#thoughtText");
const eventText = document.querySelector("#eventText");
const actionText = document.querySelector("#actionText");
const inventoryText = document.querySelector("#inventoryText");
const entityList = document.querySelector("#entityList");
const knownMap = document.querySelector("#knownMap");

const replay = await loadReplay();
const frames = replay.frames;
let frameIndex = 0;
let playing = true;
let frameAccumulator = 0;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x070a12, 0.035);

const camera = new THREE.PerspectiveCamera(54, 1, 0.1, 140);
camera.position.set(-9, 9, 11);

const clock = new THREE.Clock();
const cellSize = replay.scene.cell_size_m;
const wallHeight = replay.scene.wall_height_m;
const objects = new Map();
const scanGroup = new THREE.Group();
const moteGroup = new THREE.Group();

scene.add(scanGroup);
scene.add(moteGroup);
buildLighting();
buildVault();
buildPath();
buildAgent();
buildMotes();

timeline.max = String(frames.length - 1);
timeline.addEventListener("input", () => {
  frameIndex = Number(timeline.value);
  playing = false;
  playButton.textContent = "Play";
  renderFrame(frameIndex, true);
});

playButton.addEventListener("click", () => {
  playing = !playing;
  playButton.textContent = playing ? "Pause" : "Play";
});

window.addEventListener("resize", resize);
resize();
renderFrame(0, true);
renderer.setAnimationLoop(animate);

async function loadReplay() {
  const response = await fetch("../examples/demo_replay.json");
  if (!response.ok) {
    throw new Error(`Could not load replay JSON: ${response.status}`);
  }
  return response.json();
}

function buildLighting() {
  const ambient = new THREE.HemisphereLight(0x9ab7ff, 0x08080c, 1.2);
  scene.add(ambient);

  const key = new THREE.DirectionalLight(0xfff2cb, 2.2);
  key.position.set(-8, 16, -4);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  scene.add(key);

  const cyan = new THREE.PointLight(0x67e8f9, 2.4, 28);
  cyan.position.set(2, 4, 4);
  scene.add(cyan);

  const green = new THREE.PointLight(0x20f5a6, 3.2, 30);
  const goal = replay.scene.goal.center_m;
  green.position.set(goal.x, 2.6, goal.z);
  scene.add(green);
}

function buildVault() {
  const layout = replay.scene.layout;
  const floorMaterial = new THREE.MeshStandardMaterial({
    color: 0x171b24,
    roughness: 0.78,
    metalness: 0.22,
  });
  const wallMaterial = new THREE.MeshStandardMaterial({
    color: 0x222938,
    roughness: 0.62,
    metalness: 0.34,
  });
  const trimMaterial = new THREE.MeshStandardMaterial({
    color: 0x7f8cff,
    emissive: 0x20265c,
    metalness: 0.75,
    roughness: 0.28,
  });

  for (let z = 0; z < layout.length; z += 1) {
    for (let x = 0; x < layout[z].length; x += 1) {
      const world = gridToWorld(x, z, 0);
      const floor = new THREE.Mesh(
        new THREE.BoxGeometry(cellSize * 0.98, 0.08, cellSize * 0.98),
        floorMaterial,
      );
      floor.position.set(world.x, -0.04, world.z);
      floor.receiveShadow = true;
      scene.add(floor);

      if (layout[z][x] === "#") {
        const wall = new THREE.Mesh(
          new THREE.BoxGeometry(cellSize, wallHeight, cellSize),
          wallMaterial,
        );
        wall.position.set(world.x, wallHeight / 2, world.z);
        wall.castShadow = true;
        wall.receiveShadow = true;
        scene.add(wall);

        const trim = new THREE.Mesh(
          new THREE.BoxGeometry(cellSize * 0.82, 0.04, cellSize * 0.82),
          trimMaterial,
        );
        trim.position.set(world.x, wallHeight + 0.035, world.z);
        scene.add(trim);
      }

      if (layout[z][x] === "D") {
        const door = createDoor(x, z);
        objects.set(`door-${x}-${z}`, door);
        scene.add(door);
      }

      if (layout[z][x] === "K") {
        const key = createKey(x, z);
        objects.set(`brass_key-${x}-${z}`, key);
        scene.add(key);
      }

      if (layout[z][x] === "G") {
        scene.add(createGoal(x, z));
      }
    }
  }

  for (const decoration of replay.scene.decorations) {
    scene.add(createLightPillar(decoration.center_m));
  }
}

function createDoor(x, z) {
  const group = new THREE.Group();
  const center = gridToWorld(x, z, 0);
  group.position.set(center.x, 0, center.z);

  const panel = new THREE.Mesh(
    new THREE.BoxGeometry(0.24, 2.85, cellSize * 0.86),
    new THREE.MeshStandardMaterial({
      color: 0x236bff,
      emissive: 0x0d3ed1,
      transparent: true,
      opacity: 0.68,
      roughness: 0.2,
      metalness: 0.1,
    }),
  );
  panel.position.y = 1.42;
  panel.castShadow = true;
  group.add(panel);

  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.66, 0.035, 12, 48),
    new THREE.MeshStandardMaterial({
      color: 0x67e8f9,
      emissive: 0x236bff,
      metalness: 0.8,
      roughness: 0.18,
    }),
  );
  ring.position.y = 1.55;
  ring.rotation.y = Math.PI / 2;
  group.add(ring);

  group.userData.panel = panel;
  group.userData.ring = ring;
  return group;
}

function createKey(x, z) {
  const group = new THREE.Group();
  const center = gridToWorld(x, z, 0);
  group.position.set(center.x, 0.28, center.z);

  const pedestal = new THREE.Mesh(
    new THREE.CylinderGeometry(0.48, 0.62, 0.46, 8),
    new THREE.MeshStandardMaterial({ color: 0x5b4631, metalness: 0.45, roughness: 0.42 }),
  );
  pedestal.castShadow = true;
  pedestal.receiveShadow = true;
  group.add(pedestal);

  const key = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color: 0xf5c542,
    emissive: 0x8a5b00,
    metalness: 0.9,
    roughness: 0.18,
  });
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.22, 0.035, 16, 36), material);
  ring.position.set(-0.24, 0.62, 0);
  const shaft = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.07, 0.07), material);
  shaft.position.set(0.1, 0.62, 0);
  const tooth = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.22, 0.07), material);
  tooth.position.set(0.38, 0.52, 0);
  key.add(ring, shaft, tooth);
  key.rotation.z = -0.2;
  group.add(key);
  group.userData.artifact = key;
  return group;
}

function createGoal(x, z) {
  const group = new THREE.Group();
  const center = gridToWorld(x, z, 0);
  group.position.set(center.x, 0, center.z);

  const beam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.64, 0.9, 3.4, 48, 1, true),
    new THREE.MeshStandardMaterial({
      color: 0x20f5a6,
      emissive: 0x0d7a57,
      transparent: true,
      opacity: 0.33,
      side: THREE.DoubleSide,
    }),
  );
  beam.position.y = 1.72;
  group.add(beam);

  const core = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.46, 1),
    new THREE.MeshStandardMaterial({
      color: 0xb7ffe5,
      emissive: 0x20f5a6,
      metalness: 0.25,
      roughness: 0.1,
    }),
  );
  core.position.y = 1.45;
  group.add(core);
  group.userData.core = core;
  return group;
}

function createLightPillar(center) {
  const group = new THREE.Group();
  group.position.set(center.x, 0, center.z);
  const column = new THREE.Mesh(
    new THREE.CylinderGeometry(0.12, 0.18, 2.8, 18),
    new THREE.MeshStandardMaterial({
      color: 0x7f8cff,
      emissive: 0x20265c,
      transparent: true,
      opacity: 0.42,
    }),
  );
  column.position.y = 1.4;
  group.add(column);
  const light = new THREE.PointLight(0x7f8cff, 1.2, 7);
  light.position.y = 2.2;
  group.add(light);
  return group;
}

function buildPath() {
  const points = frames.map((frame) => {
    const pos = frame.agent.pose.position_m;
    return new THREE.Vector3(pos.x, 0.08, pos.z);
  });
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color: 0xff6b35, transparent: true, opacity: 0.72 });
  const line = new THREE.Line(geometry, material);
  scene.add(line);
}

function buildAgent() {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.33, 0.76, 6, 14),
    new THREE.MeshStandardMaterial({
      color: 0xff6b35,
      emissive: 0x5f1606,
      metalness: 0.28,
      roughness: 0.34,
    }),
  );
  body.position.y = 0.72;
  body.castShadow = true;
  group.add(body);

  const visor = new THREE.Mesh(
    new THREE.BoxGeometry(0.5, 0.12, 0.08),
    new THREE.MeshStandardMaterial({ color: 0x67e8f9, emissive: 0x0e7490 }),
  );
  visor.position.set(0, 1.18, -0.26);
  group.add(visor);

  const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 1.35, 0), 1.1, 0xfff2cb, 0.32, 0.18);
  group.add(arrow);
  objects.set("agent", group);
  scene.add(group);
}

function buildMotes() {
  const material = new THREE.MeshBasicMaterial({ color: 0x67e8f9, transparent: true, opacity: 0.42 });
  const geometry = new THREE.SphereGeometry(0.025, 8, 8);
  for (let i = 0; i < 120; i += 1) {
    const mote = new THREE.Mesh(geometry, material);
    mote.position.set(
      (Math.random() - 0.5) * 22,
      0.6 + Math.random() * 3.2,
      (Math.random() - 0.5) * 14,
    );
    mote.userData.phase = Math.random() * Math.PI * 2;
    moteGroup.add(mote);
  }
}

function renderFrame(index, snap = false) {
  const frame = frames[index];
  const agent = objects.get("agent");
  const pose = frame.agent.pose;
  const position = pose.position_m;
  const yaw = THREE.MathUtils.degToRad(pose.yaw_degrees);

  const targetPosition = new THREE.Vector3(position.x, 0, position.z);
  if (snap) {
    agent.position.copy(targetPosition);
  } else {
    agent.position.lerp(targetPosition, 0.13);
  }
  agent.rotation.y = yaw;

  updateDoorState(frame);
  updateItemState(frame);
  updateScanLines(frame);
  updateHud(frame, index);
  updateCamera(agent.position, pose.yaw_degrees, snap);
}

function updateDoorState(frame) {
  for (const door of frame.scene_state.doors) {
    const object = objects.get(door.id);
    if (!object) continue;
    const panel = object.userData.panel;
    const ring = object.userData.ring;
    panel.visible = !door.open;
    panel.material.color.set(door.locked ? 0x236bff : 0x67e8f9);
    panel.material.emissive.set(door.locked ? 0x0d3ed1 : 0x0e7490);
    ring.rotation.z += 0.025;
    ring.material.emissiveIntensity = door.open ? 2.8 : 1.2;
  }
}

function updateItemState(frame) {
  const presentItems = new Set(frame.scene_state.items.map((item) => item.id));
  for (const [id, object] of objects.entries()) {
    if (!id.startsWith("brass_key-")) continue;
    object.visible = presentItems.has(id);
    if (object.visible && object.userData.artifact) {
      object.userData.artifact.rotation.y += 0.045;
      object.userData.artifact.position.y = Math.sin(performance.now() / 360) * 0.06;
    }
  }
}

function updateScanLines(frame) {
  scanGroup.clear();
  const origin = frame.agent.pose.position_m;
  for (const entity of frame.visible_entities.slice(0, 12)) {
    const material = new THREE.LineBasicMaterial({
      color: entity.blocks_movement ? 0x7f8cff : 0x20f5a6,
      transparent: true,
      opacity: 0.34,
    });
    const target = entity.center_m;
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(origin.x, 0.35, origin.z),
      new THREE.Vector3(target.x, Math.max(0.35, target.y), target.z),
    ]);
    scanGroup.add(new THREE.Line(geometry, material));
  }
}

function updateHud(frame, index) {
  timeline.value = String(index);
  const result = replay.result.success ? "success" : "failure";
  resultBadge.textContent = `${result} in ${replay.result.steps} steps`;
  stepLabel.textContent = `Step ${frame.step}`;
  const pose = frame.agent.pose.position_m;
  poseLabel.textContent = `(${pose.x}, ${pose.y}, ${pose.z})m | yaw ${frame.agent.pose.yaw_degrees}`;
  thoughtText.textContent = frame.thought || "Initial scene scan.";
  eventText.textContent = frame.event;
  actionText.textContent = frame.action ? frame.action.type : "spawn";
  inventoryText.textContent = frame.agent.inventory.length ? frame.agent.inventory.join(", ") : "[]";
  knownMap.textContent = frame.known_map;

  entityList.replaceChildren();
  for (const entity of frame.visible_entities.slice(0, 7)) {
    const item = document.createElement("li");
    item.textContent = `${entity.label} | ${entity.distance_m}m | ${entity.bearing_degrees} deg`;
    entityList.appendChild(item);
  }
  if (!entityList.children.length) {
    const item = document.createElement("li");
    item.textContent = "No salient entities in the current field scan.";
    entityList.appendChild(item);
  }
}

function updateCamera(agentPosition, yawDegrees, snap) {
  const direction = yawToVector(yawDegrees);
  const desired = new THREE.Vector3(
    agentPosition.x - direction.x * 6.2,
    5.1,
    agentPosition.z - direction.z * 6.2,
  );
  desired.x += 2.2;
  const target = new THREE.Vector3(agentPosition.x, 1.05, agentPosition.z);
  if (snap) {
    camera.position.copy(desired);
  } else {
    camera.position.lerp(desired, 0.055);
  }
  camera.lookAt(target);
}

function animate() {
  const delta = clock.getDelta();
  if (playing) {
    frameAccumulator += delta * Number(speed.value);
    if (frameAccumulator > 0.52) {
      frameAccumulator = 0;
      frameIndex = (frameIndex + 1) % frames.length;
    }
  }

  const time = performance.now() / 1000;
  for (const mote of moteGroup.children) {
    mote.position.y += Math.sin(time + mote.userData.phase) * 0.0009;
  }

  renderFrame(frameIndex);
  renderer.render(scene, camera);
}

function resize() {
  const box = viewport.getBoundingClientRect();
  renderer.setSize(box.width, box.height);
  camera.aspect = box.width / box.height;
  camera.updateProjectionMatrix();
}

function gridToWorld(x, z, y) {
  const width = replay.scene.layout[0].length;
  const depth = replay.scene.layout.length;
  return {
    x: (x - (width - 1) / 2) * cellSize,
    y,
    z: (z - (depth - 1) / 2) * cellSize,
  };
}

function yawToVector(degrees) {
  const radians = THREE.MathUtils.degToRad(degrees);
  return new THREE.Vector3(Math.sin(radians), 0, -Math.cos(radians));
}
