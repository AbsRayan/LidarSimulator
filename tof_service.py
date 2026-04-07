import sys
import os
import numpy as np
from scene_state import SceneState
from camera_controller import CameraController
from geometry_utils import build_rotation_matrix, compute_mesh_normalization, transform_vertex

class ToFService:
    def calculate_tof(self, scene_state: SceneState, camera_controller: CameraController, airplane_mesh):
        if not airplane_mesh:
            return
            
        try:
            tof_path = os.path.join(os.path.dirname(__file__), 'tof_camera')
            old_path = list(sys.path)
            old_geometry = sys.modules.pop('geometry', None)
            
            try:
                sys.path.insert(0, tof_path)
                from geometry import Point as ToFPoint, Triangle as ToFTriangle, Figure as ToFFigure
                from tof_modeling import ToFCamera
            finally:
                sys.path = old_path
                sys.modules.pop('geometry', None)
                if old_geometry is not None:
                    sys.modules['geometry'] = old_geometry
            
            all_points = airplane_mesh.vectors.reshape(-1, 3)
            center, scale = compute_mesh_normalization(all_points)
            
            rot_matrix = build_rotation_matrix(scene_state.airplane_rot[0], scene_state.airplane_rot[1], scene_state.airplane_rot[2])
            offset = np.array(scene_state.airplane_pos)
            
            triangles = []
            for vec in airplane_mesh.vectors:
                try:
                    v1 = transform_vertex(vec[0], rot_matrix, offset, center, scale)
                    v2 = transform_vertex(vec[1], rot_matrix, offset, center, scale)
                    v3 = transform_vertex(vec[2], rot_matrix, offset, center, scale)
                    
                    p1 = ToFPoint(v1)
                    p2 = ToFPoint(v2)
                    p3 = ToFPoint(v3)
                    triangles.append(ToFTriangle(p1, p2, p3))
                except ValueError:
                    pass
            
            if not triangles:
                return
                
            figure = ToFFigure(triangles=triangles, use_octree=True)
            
            direction = np.array(scene_state.tof_dir)
            if np.linalg.norm(direction) < 1e-5:
                direction = np.array([0.0, 0.0, -1.0])
                
            cam = ToFCamera(
                position=ToFPoint(np.array(scene_state.tof_pos)),
                width=100,
                height=100,
                direction=direction,
                fov=camera_controller.fov
            )
            
            cam.get_points_and_distances_to_object(figure, parallel=False, use_octree=True)
            scene_state.tof_distances = cam.object_distances.copy()
            scene_state.tof_resolution = (100, 100)
            
            scene_state.tof_points = cam.object_points if cam.object_points is not None else np.array([])
            
        except ImportError as e:
            print(f"Ошибка загрузки модулей ToF: {e}")
        except Exception as e:
            print(f"Ошибка расчёта ToF: {e}")

    def save_depth_map(self, scene_state: SceneState, filename="depth_map.png"):
        if not hasattr(scene_state, 'tof_distances') or scene_state.tof_distances is None:
            return False

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        width, height = scene_state.tof_resolution
        depth_map = scene_state.tof_distances.reshape((width, height))

        figure, axis = plt.subplots(figsize=(8, 6))

        if np.all(np.isnan(depth_map)):
            masked_depth = np.ma.masked_where(np.ones_like(depth_map, dtype=bool), depth_map)
            axis.imshow(masked_depth, cmap='plasma_r')
        else:
            axis.imshow(depth_map, cmap='plasma_r', vmin=np.nanmin(depth_map), vmax=np.nanmax(depth_map))

        axis.axis("image")
        axis.grid(False)
        axis.axis("off")

        plt.savefig(filename)
        plt.close(figure)
        print(f"Карта глубин сохранена в {filename}")
        return True
