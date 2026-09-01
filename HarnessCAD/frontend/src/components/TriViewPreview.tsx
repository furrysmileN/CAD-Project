import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

type Props = {
  modelUrl: string;
};

type ViewId = "x" | "y" | "z";

type ViewConfig = {
  id: ViewId;
  label: string;
};

const VIEW_CONFIGS: ViewConfig[] = [
  { id: "x", label: "X轴视图 (YZ平面)" },
  { id: "y", label: "Y轴视图 (XZ平面)" },
  { id: "z", label: "Z轴视图 (XY平面)" }
];

function createLabelSprite(text: string, scale: number): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 300;
  canvas.height = 100;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return new THREE.Sprite();
  }
  ctx.fillStyle = "rgba(15, 23, 42, 0.9)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f8fafc";
  ctx.font = "bold 30px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, depthTest: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(scale * 2.6, scale * 0.9, 1);
  return sprite;
}

function addDimLine(
  group: THREE.Group,
  start: THREE.Vector3,
  end: THREE.Vector3,
  label: string,
  scale: number
): void {
  const lineMat = new THREE.LineBasicMaterial({ color: 0xb91c1c });
  const lineGeom = new THREE.BufferGeometry().setFromPoints([start, end]);
  const line = new THREE.Line(lineGeom, lineMat);
  group.add(line);

  const labelSprite = createLabelSprite(label, scale);
  labelSprite.position.copy(start.clone().add(end).multiplyScalar(0.5));
  group.add(labelSprite);
}

export default function TriViewPreview({ modelUrl }: Props) {
  const xRef = useRef<HTMLDivElement>(null);
  const yRef = useRef<HTMLDivElement>(null);
  const zRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState("");
  const renderersRef = useRef<Record<ViewId, THREE.WebGLRenderer | null>>({
    x: null,
    y: null,
    z: null
  });

  useEffect(() => {
    if (!modelUrl || !xRef.current || !yRef.current || !zRef.current) {
      return;
    }

    setError("");

    const containers: Record<ViewId, HTMLDivElement> = {
      x: xRef.current,
      y: yRef.current,
      z: zRef.current
    };

    const disposables: Array<() => void> = [];
    const loader = new STLLoader();

    loader.load(
      modelUrl,
      (geometry) => {
        geometry.computeVertexNormals();
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        if (!box) {
          setError("三视图生成失败：无法读取模型边界。");
          return;
        }

        const center = new THREE.Vector3();
        box.getCenter(center);

        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        const halfX = size.x / 2;
        const halfY = size.y / 2;
        const halfZ = size.z / 2;

        for (const cfg of VIEW_CONFIGS) {
          const container = containers[cfg.id];
          const width = container.clientWidth || 320;
          const height = container.clientHeight || 240;
          const aspect = width / height;
          const frustum = maxDim * 1.6;

          const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
          renderer.setSize(width, height);
          renderer.setPixelRatio(window.devicePixelRatio);
          renderer.setClearColor(0xffffff, 1);
          container.innerHTML = "";
          container.appendChild(renderer.domElement);
          renderersRef.current[cfg.id] = renderer;

          const scene = new THREE.Scene();
          scene.background = new THREE.Color(0xffffff);
          scene.add(new THREE.AmbientLight(0xffffff, 1));

          const camera = new THREE.OrthographicCamera(
            (-frustum * aspect) / 2,
            (frustum * aspect) / 2,
            frustum / 2,
            -frustum / 2,
            0.1,
            4000
          );

          if (cfg.id === "x") {
            camera.position.set(maxDim * 2, 0, 0);
            camera.up.set(0, 0, 1);
          } else if (cfg.id === "y") {
            camera.position.set(0, maxDim * 2, 0);
            camera.up.set(0, 0, 1);
          } else {
            camera.position.set(0, 0, maxDim * 2);
            camera.up.set(0, 1, 0);
          }
          camera.lookAt(0, 0, 0);

          const meshMaterial = new THREE.MeshBasicMaterial({
            color: 0x60a5fa,
            transparent: true,
            opacity: 0.12,
            depthWrite: false
          });
          const mesh = new THREE.Mesh(geometry.clone(), meshMaterial);
          mesh.position.set(-center.x, -center.y, -center.z);
          scene.add(mesh);

          const edgeGeometry = new THREE.EdgesGeometry(mesh.geometry as THREE.BufferGeometry);
          const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x0f172a, transparent: true, opacity: 0.9 });
          const edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial);
          edgeLines.position.copy(mesh.position);
          scene.add(edgeLines);

          const dimGroup = new THREE.Group();
          const offset = Math.max(maxDim * 0.22, 3);
          // Keep labels proportional to the model. A fixed 1.8-world-unit
          // minimum overwhelms normalized CAD whose longest edge is about 1.
          const labelScale = Math.max(maxDim * 0.1, 0.0001);
          if (cfg.id === "x") {
            addDimLine(
              dimGroup,
              new THREE.Vector3(halfX + offset, -halfY, -halfZ),
              new THREE.Vector3(halfX + offset, halfY, -halfZ),
              `Y=${size.y.toFixed(2)}`,
              labelScale
            );
            addDimLine(
              dimGroup,
              new THREE.Vector3(halfX + offset, halfY, -halfZ),
              new THREE.Vector3(halfX + offset, halfY, halfZ),
              `Z=${size.z.toFixed(2)}`,
              labelScale
            );
          } else if (cfg.id === "y") {
            addDimLine(
              dimGroup,
              new THREE.Vector3(-halfX, halfY + offset, -halfZ),
              new THREE.Vector3(halfX, halfY + offset, -halfZ),
              `X=${size.x.toFixed(2)}`,
              labelScale
            );
            addDimLine(
              dimGroup,
              new THREE.Vector3(halfX, halfY + offset, -halfZ),
              new THREE.Vector3(halfX, halfY + offset, halfZ),
              `Z=${size.z.toFixed(2)}`,
              labelScale
            );
          } else {
            addDimLine(
              dimGroup,
              new THREE.Vector3(-halfX, -halfY, halfZ + offset),
              new THREE.Vector3(halfX, -halfY, halfZ + offset),
              `X=${size.x.toFixed(2)}`,
              labelScale
            );
            addDimLine(
              dimGroup,
              new THREE.Vector3(halfX, -halfY, halfZ + offset),
              new THREE.Vector3(halfX, halfY, halfZ + offset),
              `Y=${size.y.toFixed(2)}`,
              labelScale
            );
          }
          scene.add(dimGroup);

          renderer.render(scene, camera);

          disposables.push(() => {
            renderer.dispose();
            mesh.geometry.dispose();
            meshMaterial.dispose();
            edgeGeometry.dispose();
            edgeMaterial.dispose();
            dimGroup.traverse((obj) => {
              const withGeometry = obj as THREE.Object3D & { geometry?: THREE.BufferGeometry };
              const withMaterial = obj as THREE.Object3D & {
                material?: THREE.Material | THREE.Material[];
              };
              if (withGeometry.geometry) {
                withGeometry.geometry.dispose();
              }
              if (withMaterial.material) {
                if (Array.isArray(withMaterial.material)) {
                  withMaterial.material.forEach((item) => item.dispose());
                } else {
                  withMaterial.material.dispose();
                }
              }
            });
          });
        }
      },
      undefined,
      () => {
        setError("三视图生成失败，请稍后重试。");
      }
    );

    return () => {
      for (const dispose of disposables) {
        dispose();
      }
      renderersRef.current = { x: null, y: null, z: null };
    };
  }, [modelUrl]);

  const handleExport = () => {
    for (const cfg of VIEW_CONFIGS) {
      const renderer = renderersRef.current[cfg.id];
      if (!renderer) {
        continue;
      }
      const url = renderer.domElement.toDataURL("image/png");
      const a = document.createElement("a");
      a.href = url;
      a.download = `tri_view_${cfg.id}.png`;
      a.click();
    }
  };

  return (
    <div className="tri-view-wrap">
      <div className="tri-view-head">
        <h3>自动三视图（带尺寸）</h3>
        <button onClick={handleExport}>导出三视图 PNG</button>
      </div>
      <div className="tri-view-grid">
        <div className="tri-view-item">
          <p className="hint">{VIEW_CONFIGS[0].label}</p>
          <div className="tri-canvas" ref={xRef} />
        </div>
        <div className="tri-view-item">
          <p className="hint">{VIEW_CONFIGS[1].label}</p>
          <div className="tri-canvas" ref={yRef} />
        </div>
        <div className="tri-view-item">
          <p className="hint">{VIEW_CONFIGS[2].label}</p>
          <div className="tri-canvas" ref={zRef} />
        </div>
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
