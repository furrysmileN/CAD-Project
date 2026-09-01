import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

type Mode = "side" | "overlay";

type Props = {
  gtUrl: string | null;
  predUrl: string | null;
  mode: Mode;
};

const GT_COLOR = 0x2f6f9f;
const PRED_COLOR = 0xc45c26;

function fitGeometry(geometry: THREE.BufferGeometry, targetSize = 42) {
  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  if (!box) {
    return;
  }
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  geometry.translate(-center.x, -center.y, -center.z);
  const longest = Math.max(size.x, size.y, size.z) || 1;
  geometry.scale(targetSize / longest, targetSize / longest, targetSize / longest);
}

function makeMesh(geometry: THREE.BufferGeometry, color: number, opacity: number) {
  const material = new THREE.MeshPhongMaterial({
    color,
    transparent: opacity < 1,
    opacity,
    shininess: 18,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry, 25),
    new THREE.LineBasicMaterial({ color: 0x0f172a, transparent: true, opacity: 0.35 })
  );
  mesh.add(edges);
  return mesh;
}

function mountScene(container: HTMLDivElement) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f6fa);
  const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 2000);
  camera.position.set(70, 48, 70);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x64748b, 1.15));
  const key = new THREE.DirectionalLight(0xffffff, 0.85);
  key.position.set(60, 80, 40);
  scene.add(key);
  scene.add(new THREE.GridHelper(80, 16, 0xcbd5e1, 0xe2e8f0));
  return { scene, camera, renderer, controls };
}

export default function DualCadViewer({ gtUrl, predUrl, mode }: Props) {
  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const loaders: Array<() => void> = [];
    const loader = new STLLoader();
    let disposed = false;

    const load = (url: string | null) =>
      new Promise<THREE.BufferGeometry | null>((resolve) => {
        if (!url) {
          resolve(null);
          return;
        }
        loader.load(
          url,
          (geometry) => {
            fitGeometry(geometry);
            resolve(geometry);
          },
          undefined,
          () => resolve(null)
        );
      });

    const start = async () => {
      const [gtGeom, predGeom] = await Promise.all([load(gtUrl), load(predUrl)]);
      if (disposed) {
        gtGeom?.dispose();
        predGeom?.dispose();
        return;
      }

      const syncPair: OrbitControls[] = [];
      const bindSync = (controls: OrbitControls) => {
        syncPair.push(controls);
        controls.addEventListener("change", () => {
          for (const other of syncPair) {
            if (other === controls) {
              continue;
            }
            other.object.position.copy(controls.object.position);
            other.object.quaternion.copy(controls.object.quaternion);
            other.target.copy(controls.target);
          }
        });
      };

      const attach = (container: HTMLDivElement | null, geometry: THREE.BufferGeometry | null, color: number, opacity: number) => {
        if (!container) {
          return;
        }
        const { scene, camera, renderer, controls } = mountScene(container);
        if (geometry) {
          scene.add(makeMesh(geometry, color, opacity));
        }
        bindSync(controls);
        let frame = 0;
        const tick = () => {
          frame = requestAnimationFrame(tick);
          controls.update();
          renderer.render(scene, camera);
        };
        tick();
        const onResize = () => {
          camera.aspect = container.clientWidth / container.clientHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(container.clientWidth, container.clientHeight);
        };
        window.addEventListener("resize", onResize);
        loaders.push(() => {
          cancelAnimationFrame(frame);
          window.removeEventListener("resize", onResize);
          controls.dispose();
          renderer.dispose();
          renderer.domElement.remove();
        });
      };

      if (mode === "overlay") {
        const container = overlayRef.current;
        if (!container) {
          return;
        }
        const { scene, camera, renderer, controls } = mountScene(container);
        if (gtGeom) {
          scene.add(makeMesh(gtGeom, GT_COLOR, 0.42));
        }
        if (predGeom) {
          scene.add(makeMesh(predGeom, PRED_COLOR, 0.55));
        }
        let frame = 0;
        const tick = () => {
          frame = requestAnimationFrame(tick);
          controls.update();
          renderer.render(scene, camera);
        };
        tick();
        const onResize = () => {
          camera.aspect = container.clientWidth / container.clientHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(container.clientWidth, container.clientHeight);
        };
        window.addEventListener("resize", onResize);
        loaders.push(() => {
          cancelAnimationFrame(frame);
          window.removeEventListener("resize", onResize);
          controls.dispose();
          renderer.dispose();
          renderer.domElement.remove();
          gtGeom?.dispose();
          predGeom?.dispose();
        });
        return;
      }

      attach(leftRef.current, gtGeom, GT_COLOR, 1);
      attach(rightRef.current, predGeom, PRED_COLOR, 1);
      loaders.push(() => {
        gtGeom?.dispose();
        predGeom?.dispose();
      });
    };

    void start();
    return () => {
      disposed = true;
      for (const dispose of loaders) {
        dispose();
      }
    };
  }, [gtUrl, predUrl, mode]);

  if (mode === "overlay") {
    return <div className="compare-viewer overlay" ref={overlayRef} />;
  }
  return (
    <div className="compare-split">
      <div>
        <div className="compare-caption gt">真值 Ground Truth</div>
        <div className="compare-viewer" ref={leftRef} />
      </div>
      <div>
        <div className="compare-caption pred">模型生成</div>
        <div className="compare-viewer" ref={rightRef} />
      </div>
    </div>
  );
}
