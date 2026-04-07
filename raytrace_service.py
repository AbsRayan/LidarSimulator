import os
import numpy as np
from PIL.Image import Image

from scene_state import SceneState
from camera_controller import CameraController
from geometry_utils import build_rotation_matrix, transform_vertex, compute_mesh_normalization

class RaytraceService:
    def calculate_raytrace(self, scene_state: SceneState, camera_controller: CameraController, width: int = 400, height: int = 300) -> Image:
        """
        Runs a raytrace render and returns the resulting PIL Image.
        """
        try:
            from stl_parser import parse_binary_stl
            from core.camera import Camera as RayCamera
            from core.light import Light
            from core.plane import Plane
            from core.material import Material
            from renderer.mesh_renderer import MeshRenderer
            from geometry.mesh import Mesh, Triangle as RayTriangle
            from PIL import Image as PilImage

            stl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'Mig29.stl')
            airplane_color = np.array([185, 185, 185], dtype=np.float64)

            print("[Raytracer] Загрузка STL...")
            raw_mesh = parse_binary_stl(stl_path, airplane_color)

            all_verts = np.array([[t.v0, t.v1, t.v2] for t in raw_mesh.triangles]).reshape(-1, 3)
            center, scale = compute_mesh_normalization(all_verts)

            airplane_mat = Material(
                color=airplane_color,
                diffuse=0.9,
                specular=0.3,
                shininess=50.0,
                reflection=0.1
            )

            def _add_airplane(tris, rot_m, offs):
                for tri in raw_mesh.triangles:
                    v0 = transform_vertex(tri.v0, rot_m, offs, center, scale)
                    v1 = transform_vertex(tri.v1, rot_m, offs, center, scale)
                    v2 = transform_vertex(tri.v2, rot_m, offs, center, scale)
                    n = rot_m @ tri.normal
                    nl = np.linalg.norm(n)
                    if nl < 1e-6:
                        continue
                    tris.append(RayTriangle(v0, v1, v2, n / nl, airplane_color, airplane_mat))

            all_triangles = []

            print("[Raytracer] Добавление главного самолёта...")
            rot_main = build_rotation_matrix(scene_state.airplane_rot[0], scene_state.airplane_rot[1], scene_state.airplane_rot[2])
            _add_airplane(all_triangles, rot_main, np.array(scene_state.airplane_pos, dtype=np.float64))

            print("[Raytracer] Добавление 5 припаркованных самолётов...")
            rot_parked = build_rotation_matrix(-90.0, 0.0, 0.0)
            for i in range(5):
                _add_airplane(all_triangles, rot_parked, np.array([18.0 + i * 2.5, 0.0, -7.5]))

            ground_color = np.array([70, 100, 60], dtype=np.float64)
            ground_mat = Material(color=ground_color, diffuse=0.85)
            gn = np.array([0.0, 1.0, 0.0])
            gs, gy = 100.0, -0.3
            all_triangles.append(RayTriangle(np.array([-gs,gy,-gs]), np.array([gs,gy,-gs]), np.array([gs,gy,gs]),  gn, ground_color, ground_mat))
            all_triangles.append(RayTriangle(np.array([-gs,gy,-gs]), np.array([gs,gy, gs]), np.array([-gs,gy,gs]), gn, ground_color, ground_mat))

            if not all_triangles:
                print("[Raytracer] Нет треугольников.")
                return None

            print(f"[Raytracer] Строим BVH для {len(all_triangles)} треугольников...")
            scene_mesh = Mesh(all_triangles)

            cam_pos = np.array(camera_controller.pos, dtype=np.float64)
            cam_target = np.array(camera_controller.target, dtype=np.float64)

            camera = RayCamera(
                position=cam_pos,
                look_at=cam_target,
                vector_up=np.array([0.0, 1.0, 0.0]),
                fov_vertical=camera_controller.fov,
                fov_horizontal=camera_controller.fov * (width / height)
            )

            light = Light(
                position=np.array([10.0, 20.0, 10.0]),
                color=np.array([255, 255, 220], dtype=np.uint8),
                intensity=2.0
            )

            plane = Plane(
                point=np.array([0.0, -0.3, 0.0]),
                normal=np.array([0.0, 1.0, 0.0]),
                color=np.array([80, 100, 80], dtype=np.float64)
            )

            renderer = MeshRenderer(
                mesh=scene_mesh,
                camera=camera,
                plane=plane,
                light=light
            )

            print(f"[Raytracer] Рендеринг {width}x{height}...")
            image_array = renderer.render(width, height)

            img = PilImage.fromarray(image_array.astype(np.uint8))
            return img

        except Exception as e:
            import traceback
            print(f"[Raytracer] Ошибка: {e}")
            traceback.print_exc()
            return None
