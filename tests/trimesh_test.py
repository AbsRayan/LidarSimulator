import time
import numpy as np
import trimesh
from core.camera import Camera
from core.plane import Plane
from core.light import Light
from core.material import Material
from geometry.mesh import Mesh, Triangle
from renderer.mesh_renderer import MeshRenderer
from PIL import Image
import os

def test_trimesh_render():
    print("="*60)
    print("ТЕСТ TRIMESH С САМОЛЕТОМ")
    print("="*60)
    
    stl_file = "Mig29.stl"
    if not os.path.exists(stl_file):
        print(f"Файл {stl_file} не найден")
        return
    
    mesh_trimesh = trimesh.load(stl_file)
    vertices = mesh_trimesh.vertices
    faces = mesh_trimesh.faces
    print(f"Загружено треугольников: {len(faces)}")
    
    bounds = mesh_trimesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    size = np.max(bounds[1] - bounds[0])
    
    camera = Camera(
        position=center + np.array([size, size/2, size]),
        look_at=center,
        vector_up=np.array([0, 0, 1]),
        fov_vertical=45,
        fov_horizontal=60
    )
    
    plane = Plane(
        point=np.array([center[0], bounds[0][1] - 2, center[2]]),
        normal=np.array([0, 1, 0]),
        color=np.array([100, 100, 100])
    )
    
    light = Light(
        position=center + np.array([size, size, size]),
        color=np.array([255, 255, 255]),
        intensity=3.0
    )
    
    material = Material(
        color=np.array([185, 77, 239]),
        diffuse=0.9
    )
    
    triangles = []
    for f in faces:
        a, b, c = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
            triangles.append(Triangle(a, b, c, normal, np.array([185, 77, 239]), material))
    
    mesh = Mesh(triangles)
    
    width, height = 800, 600
    image = np.full((height, width, 3), [100, 100, 100], dtype=np.uint8)
    
    ray_origins = np.array([camera.position] * width * height)
    ray_directions = []
    
    for y in range(height):
        for x in range(width):
            u = (x + 0.5) / width - 0.5
            v = 0.5 - (y + 0.5) / height
            direction = camera.forward + u * camera.viewport_width * camera.right + v * camera.viewport_height * camera.real_up
            direction = direction / np.linalg.norm(direction)
            ray_directions.append(direction)
    
    ray_directions = np.array(ray_directions)
    
    start = time.time()
    locations, index_ray, index_tri = mesh_trimesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False
    )
    intersect_time = time.time() - start
    
    print(f"Время пересечений: {intersect_time:.2f} сек")
    print(f"Найдено пересечений: {len(locations)}")
    
    for i, ray_idx in enumerate(index_ray):
        y = ray_idx // width
        x = ray_idx % width
        if 0 <= y < height and 0 <= x < width:
            image[y, x] = np.array([185, 77, 239])
    
    img = Image.fromarray(image)
    img.save('trimesh_mig29.png')
    print("Сохранено trimesh_mig29.png")

if __name__ == "__main__":
    test_trimesh_render()