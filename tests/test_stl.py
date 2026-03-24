import numpy as np
from core.camera import Camera
from core.plane import Plane
from core.light import Light
from geometry.mesh import Mesh, Triangle
from renderer.mesh_renderer import MeshRenderer
from stl_parser import parse_binary_stl
from PIL import Image
import time

def test_stl_model():
    start_time = time.time()
    
    stl_file = "Mig29.stl"
    model_color = np.array([185, 77, 239])
    
    print(f"Загрузка {stl_file}...")
    mesh = parse_binary_stl(stl_file, model_color)
    print(f"Загружено {len(mesh.triangles)} треугольников")
    
    all_verts = []
    for tri in mesh.triangles:
        all_verts.extend([tri.v0, tri.v1, tri.v2])
    all_verts = np.array(all_verts)
    
    min_coords = np.min(all_verts, axis=0)
    max_coords = np.max(all_verts, axis=0)
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords
    
    nose_pos = np.array([center[0], max_coords[1], center[2]])
    camera_distance = size[1] * 1.5
    
    camera = Camera(
        position=np.array([
            center[0] + size[0] * 0.3,
            nose_pos[1] - camera_distance,
            center[2]
        ]),
        look_at=nose_pos,
        vector_up=np.array([0, 0, 1]),
        fov_vertical=45,
        fov_horizontal=60
    )
    
    plane = Plane(
        point=np.array([center[0], min_coords[1] - 10, center[2]]),
        normal=np.array([0, 1, 0]),
        color=np.array([100, 100, 100])
    )
    
    light = Light(
        position=np.array([
            center[0] + size[0],
            center[1] + size[1],
            center[2] + size[2]
        ]),
        color=np.array([255, 255, 220]),
        intensity=2.0
    )
    
    renderer = MeshRenderer(
        mesh=mesh,
        camera=camera,
        plane=plane,
        light=light
    )
    
    print("Рендеринг...")
    render_start = time.time()
    image = renderer.render(800, 600)
    render_time = time.time() - render_start
    
    img = Image.fromarray(image)
    img.save('mig29.png')
    
    total_time = time.time() - start_time
    print(f"Рендер: {render_time:.2f} сек")
    print(f"Всего: {total_time:.2f} сек")
    print("Сохранено как mig29.png")

if __name__ == "__main__":
    test_stl_model()