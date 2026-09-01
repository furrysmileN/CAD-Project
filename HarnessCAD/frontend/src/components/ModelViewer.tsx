import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

export type RenderMode = "solid" | "wireframe" | "solid_edges";

type Props = {
  modelUrl: string;
  renderMode: RenderMode;
  showDimensions: boolean;
  preciseMeasure: boolean;
  highlightPoints?: number[][];
};

const EMPTY_HIGHLIGHT_POINTS: number[][] = [];

type Dimensions = {
  x: number;
  y: number;
  z: number;
};

type SnapKind = "vertex" | "endpoint" | "midpoint" | "free";

type SnapCandidate = {
  point: THREE.Vector3;
  kind: Exclude<SnapKind, "free">;
};

function createTextSprite(
  text: string,
  scale: number,
  textures: THREE.Texture[],
  materials: THREE.Material[]
): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 96;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return new THREE.Sprite();
  }

  ctx.fillStyle = "rgba(15, 23, 42, 0.92)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f8fafc";
  ctx.font = "bold 30px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  textures.push(texture);
  const material = new THREE.SpriteMaterial({ map: texture, depthTest: false });
  materials.push(material);
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(scale * 2.2, scale * 0.8, 1);
  return sprite;
}

export default function ModelViewer({
  modelUrl,
  renderMode,
  showDimensions,
  preciseMeasure,
  highlightPoints
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loadError, setLoadError] = useState<string>("");
  const [dimensions, setDimensions] = useState<Dimensions | null>(null);
  const [preciseDistance, setPreciseDistance] = useState<number | null>(null);
  const [snapSummary, setSnapSummary] = useState<string>("");

  const resolvedHighlightPoints = highlightPoints ?? EMPTY_HIGHLIGHT_POINTS;

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !modelUrl) {
      return;
    }

    setLoadError("");
    setDimensions(null);
    setPreciseDistance(null);
    setSnapSummary("");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf6f7fb);

    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.1,
      2000
    );
    camera.position.set(80, 80, 80);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    controls.minDistance = 5;
    controls.maxDistance = 500;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x555555, 1.2));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(100, 120, 60);
    scene.add(directionalLight);

    const grid = new THREE.GridHelper(200, 20, 0x999999, 0xcccccc);
    scene.add(grid);

    const disposalTextures: THREE.Texture[] = [];
    const disposalMaterials: THREE.Material[] = [];
    const disposalGeometries: THREE.BufferGeometry[] = [];
    const loader = new STLLoader();
    let mesh: THREE.Mesh | null = null;
    let edgeLines: THREE.LineSegments | null = null;
    let highlightCloud: THREE.Points | null = null;
    const dimensionGroup = new THREE.Group();
    const measureGroup = new THREE.Group();
    const raycaster = new THREE.Raycaster();
    const mouseNdc = new THREE.Vector2();
    const pickedPoints: THREE.Vector3[] = [];
    const pointerDownPos = { x: 0, y: 0 };
    const pickedPointKinds: SnapKind[] = [];
    const snapCandidates: SnapCandidate[] = [];
    let snapRadiusWorld = 1.2;
    scene.add(dimensionGroup);
    scene.add(measureGroup);

    const kindLabel = (kind: SnapKind) => {
      if (kind === "vertex") {
        return "顶点";
      }
      if (kind === "endpoint") {
        return "边端点";
      }
      if (kind === "midpoint") {
        return "边中点";
      }
      return "自由点";
    };

    const toDedupKey = (point: THREE.Vector3, precision: number) =>
      `${Math.round(point.x / precision)}_${Math.round(point.y / precision)}_${Math.round(point.z / precision)}`;

    const buildSnapCandidates = (
      geometry: THREE.BufferGeometry,
      translation: THREE.Vector3,
      modelScale: number
    ) => {
      snapCandidates.length = 0;
      const dedup = new Set<string>();
      const precision = Math.max(modelScale * 0.0025, 0.05);
      const addCandidate = (pointLocal: THREE.Vector3, kind: Exclude<SnapKind, "free">) => {
        const point = pointLocal.clone().add(translation);
        const key = `${kind}_${toDedupKey(point, precision)}`;
        if (dedup.has(key)) {
          return;
        }
        dedup.add(key);
        snapCandidates.push({ point, kind });
      };

      const positionAttr = geometry.getAttribute("position");
      if (positionAttr) {
        const tmp = new THREE.Vector3();
        for (let i = 0; i < positionAttr.count; i++) {
          tmp.fromBufferAttribute(positionAttr, i);
          addCandidate(tmp, "vertex");
        }
      }

      const edgesGeometry = new THREE.EdgesGeometry(geometry);
      disposalGeometries.push(edgesGeometry);
      const edgesPos = edgesGeometry.getAttribute("position");
      if (edgesPos) {
        const start = new THREE.Vector3();
        const end = new THREE.Vector3();
        for (let i = 0; i < edgesPos.count; i += 2) {
          start.fromBufferAttribute(edgesPos, i);
          end.fromBufferAttribute(edgesPos, i + 1);
          addCandidate(start, "endpoint");
          addCandidate(end, "endpoint");
          addCandidate(start.clone().add(end).multiplyScalar(0.5), "midpoint");
        }
      }

      if (renderMode === "solid_edges") {
        const edgesMaterial = new THREE.LineBasicMaterial({ color: 0x0f172a });
        disposalMaterials.push(edgesMaterial);
        edgeLines = new THREE.LineSegments(edgesGeometry, edgesMaterial);
        edgeLines.position.copy(translation);
        scene.add(edgeLines);
      }
    };

    const getSnappedPoint = (rawPoint: THREE.Vector3): { point: THREE.Vector3; kind: SnapKind } => {
      let best: SnapCandidate | null = null;
      let bestDist = Number.POSITIVE_INFINITY;
      for (const candidate of snapCandidates) {
        const dist = candidate.point.distanceTo(rawPoint);
        if (dist < bestDist) {
          bestDist = dist;
          best = candidate;
        }
      }
      if (best && bestDist <= snapRadiusWorld) {
        return { point: best.point.clone(), kind: best.kind };
      }
      return { point: rawPoint, kind: "free" };
    };

    loader.load(
      modelUrl,
      (geometry) => {
        geometry.computeVertexNormals();
        geometry.computeBoundingBox();
        const material = new THREE.MeshStandardMaterial({
          color: 0x3f6df6,
          metalness: 0.15,
          roughness: 0.55,
          wireframe: renderMode === "wireframe"
        });
        disposalMaterials.push(material);
        mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);

        const box = geometry.boundingBox;
        if (!box) {
          return;
        }

        const center = new THREE.Vector3();
        box.getCenter(center);
        mesh.position.set(-center.x, -center.y, -center.z);
        const meshPosition = mesh.position.clone();

        const size = new THREE.Vector3();
        box.getSize(size);
        setDimensions({ x: size.x, y: size.y, z: size.z });

        const maxDim = Math.max(size.x, size.y, size.z);
        camera.position.set(maxDim * 2, maxDim * 1.8, maxDim * 1.6);
        controls.target.set(0, 0, 0);
        controls.update();

        if (resolvedHighlightPoints.length > 0) {
          const finitePoints = resolvedHighlightPoints.filter(
            (point) =>
              Array.isArray(point) &&
              point.length === 3 &&
              Number.isFinite(point[0]) &&
              Number.isFinite(point[1]) &&
              Number.isFinite(point[2])
          );
          if (finitePoints.length > 0) {
            const positions = new Float32Array(finitePoints.length * 3);
            finitePoints.forEach((point, index) => {
              positions[index * 3] = point[0] + meshPosition.x;
              positions[index * 3 + 1] = point[1] + meshPosition.y;
              positions[index * 3 + 2] = point[2] + meshPosition.z;
            });
            const highlightGeometry = new THREE.BufferGeometry();
            highlightGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
            const highlightMaterial = new THREE.PointsMaterial({
              color: 0xff3b30,
              size: Math.max(maxDim * 0.02, 0.8),
              sizeAttenuation: true
            });
            disposalGeometries.push(highlightGeometry);
            disposalMaterials.push(highlightMaterial);
            highlightCloud = new THREE.Points(highlightGeometry, highlightMaterial);
            scene.add(highlightCloud);
          }
        }

        snapRadiusWorld = Math.max(maxDim * 0.04, 0.8);
        buildSnapCandidates(geometry, mesh.position, maxDim);

        if (showDimensions) {
          const halfX = size.x / 2;
          const halfY = size.y / 2;
          const halfZ = size.z / 2;
          const offset = Math.max(maxDim * 0.15, 2);
          const lineMat = new THREE.LineBasicMaterial({ color: 0xd946ef });
          disposalMaterials.push(lineMat);

          const addDimensionLine = (start: THREE.Vector3, end: THREE.Vector3, label: string) => {
            const points = [start, end];
            const lineGeom = new THREE.BufferGeometry().setFromPoints(points);
            disposalGeometries.push(lineGeom);
            const line = new THREE.Line(lineGeom, lineMat);
            dimensionGroup.add(line);

            const labelSprite = createTextSprite(
              label,
              Math.max(maxDim * 0.1, 1.8),
              disposalTextures,
              disposalMaterials
            );
            labelSprite.position.copy(start.clone().add(end).multiplyScalar(0.5));
            dimensionGroup.add(labelSprite);
          };

          addDimensionLine(
            new THREE.Vector3(-halfX, -halfY - offset, -halfZ - offset),
            new THREE.Vector3(halfX, -halfY - offset, -halfZ - offset),
            `X: ${size.x.toFixed(2)}`
          );
          addDimensionLine(
            new THREE.Vector3(halfX + offset, -halfY, -halfZ - offset),
            new THREE.Vector3(halfX + offset, halfY, -halfZ - offset),
            `Y: ${size.y.toFixed(2)}`
          );
          addDimensionLine(
            new THREE.Vector3(halfX + offset, -halfY - offset, -halfZ),
            new THREE.Vector3(halfX + offset, -halfY - offset, halfZ),
            `Z: ${size.z.toFixed(2)}`
          );
        }
      },
      undefined,
      (error) => {
        console.error(error);
        setLoadError("模型加载失败，请稍后重试。");
      }
    );

    const clearMeasureGraphics = () => {
      while (measureGroup.children.length > 0) {
        const child = measureGroup.children.pop() as THREE.Object3D | undefined;
        if (!child) {
          continue;
        }
        if ("geometry" in child && child.geometry) {
          (child.geometry as THREE.BufferGeometry).dispose();
        }
        if ("material" in child && child.material) {
          const material = child.material as THREE.Material | THREE.Material[];
          if (Array.isArray(material)) {
            material.forEach((item) => item.dispose());
          } else {
            material.dispose();
          }
        }
      }
    };

    const drawMeasureGraphics = () => {
      clearMeasureGraphics();
      if (pickedPoints.length === 0) {
        setPreciseDistance(null);
        setSnapSummary("");
        return;
      }

      const point1Color = pickedPointKinds[0] === "free" ? 0xef4444 : 0x22c55e;
      const pointMat = new THREE.MeshBasicMaterial({ color: point1Color });
      const pointGeom = new THREE.SphereGeometry(0.9, 16, 16);
      const p1 = new THREE.Mesh(pointGeom, pointMat);
      p1.position.copy(pickedPoints[0]);
      measureGroup.add(p1);

      if (pickedPoints.length < 2) {
        setPreciseDistance(null);
        setSnapSummary(`点1：${kindLabel(pickedPointKinds[0] ?? "free")}`);
        return;
      }

      const point2Color = pickedPointKinds[1] === "free" ? 0xef4444 : 0x22c55e;
      const p2Mat = new THREE.MeshBasicMaterial({ color: point2Color });
      const p2 = new THREE.Mesh(pointGeom.clone(), p2Mat);
      p2.position.copy(pickedPoints[1]);
      measureGroup.add(p2);

      const lineGeom = new THREE.BufferGeometry().setFromPoints([pickedPoints[0], pickedPoints[1]]);
      const lineMat = new THREE.LineBasicMaterial({ color: 0xef4444 });
      const line = new THREE.Line(lineGeom, lineMat);
      measureGroup.add(line);

      const distance = pickedPoints[0].distanceTo(pickedPoints[1]);
      setPreciseDistance(distance);
      setSnapSummary(
        `点1：${kindLabel(pickedPointKinds[0] ?? "free")}，点2：${kindLabel(pickedPointKinds[1] ?? "free")}`
      );
      const label = createTextSprite(
        `D: ${distance.toFixed(3)}`,
        Math.max(distance * 0.15, 1.8),
        disposalTextures,
        disposalMaterials
      );
      label.position.copy(pickedPoints[0].clone().add(pickedPoints[1]).multiplyScalar(0.5));
      measureGroup.add(label);
    };

    const onPointerDown = (event: PointerEvent) => {
      pointerDownPos.x = event.clientX;
      pointerDownPos.y = event.clientY;
    };

    const onPointerUp = (event: PointerEvent) => {
      if (!preciseMeasure || !mesh || event.button !== 0) {
        return;
      }

      const deltaX = Math.abs(event.clientX - pointerDownPos.x);
      const deltaY = Math.abs(event.clientY - pointerDownPos.y);
      if (deltaX > 4 || deltaY > 4) {
        return;
      }

      const rect = renderer.domElement.getBoundingClientRect();
      mouseNdc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouseNdc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouseNdc, camera);
      const intersects = raycaster.intersectObject(mesh, false);
      if (intersects.length === 0) {
        return;
      }

      const point = intersects[0].point.clone();
      if (pickedPoints.length >= 2) {
        pickedPoints.length = 0;
        pickedPointKinds.length = 0;
      }
      const snapped = getSnappedPoint(point);
      pickedPoints.push(snapped.point);
      pickedPointKinds.push(snapped.kind);
      drawMeasureGraphics();
    };

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointerup", onPointerUp);

    const handleResize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener("resize", handleResize);

    let animationFrameId = 0;
    const animate = () => {
      animationFrameId = window.requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.removeEventListener("resize", handleResize);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      window.cancelAnimationFrame(animationFrameId);
      controls.dispose();
      if (mesh) {
        mesh.geometry.dispose();
      }
      if (edgeLines) {
        scene.remove(edgeLines);
      }
      if (highlightCloud) {
        scene.remove(highlightCloud);
      }
      scene.remove(dimensionGroup);
      scene.remove(measureGroup);
      disposalGeometries.forEach((item) => item.dispose());
      disposalMaterials.forEach((item) => item.dispose());
      disposalTextures.forEach((item) => item.dispose());
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [modelUrl, renderMode, showDimensions, preciseMeasure, resolvedHighlightPoints]);

  return (
    <div className="viewer-wrap">
      <div ref={containerRef} className="viewer" />
      {dimensions && (
        <p className="dim-summary">
          尺寸(mm)：X={dimensions.x.toFixed(2)} / Y={dimensions.y.toFixed(2)} / Z=
          {dimensions.z.toFixed(2)}
        </p>
      )}
      {preciseMeasure && (
        <p className="dim-summary">
          精确测量：依次点击模型两点
          {preciseDistance !== null ? `，当前距离 = ${preciseDistance.toFixed(3)} mm` : ""}
        </p>
      )}
      {preciseMeasure && snapSummary && <p className="hint">吸附状态：{snapSummary}</p>}
      {loadError && <p className="error">{loadError}</p>}
      <p className="hint">操作说明：左键旋转，右键平移，滚轮缩放。测距时点击两点，支持顶点/边端点/边中点吸附。</p>
    </div>
  );
}
