import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type Props = {
  points: number[][];
};

export default function PointCloudViewer({ points }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = ref.current;
    if (!container) {
      return;
    }
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf4f6fa);
    const camera = new THREE.PerspectiveCamera(50, container.clientWidth / Math.max(container.clientHeight, 1), 0.1, 200);
    camera.position.set(1.4, 1.1, 1.4);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x64748b, 1));
    scene.add(new THREE.AxesHelper(0.6));

    const geometry = new THREE.BufferGeometry();
    const flat = new Float32Array(points.length * 3);
    points.forEach((point, index) => {
      flat[index * 3] = point[0] ?? 0;
      flat[index * 3 + 1] = point[1] ?? 0;
      flat[index * 3 + 2] = point[2] ?? 0;
    });
    geometry.setAttribute("position", new THREE.BufferAttribute(flat, 3));
    const material = new THREE.PointsMaterial({ color: 0x1d4ed8, size: 0.012, sizeAttenuation: true });
    scene.add(new THREE.Points(geometry, material));

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
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", onResize);
      controls.dispose();
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [points]);

  return <div className="compare-viewer pc" ref={ref} />;
}
