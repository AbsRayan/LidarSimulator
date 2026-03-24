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

def test_trimesh_approximation():
    print("="*60)
    print("ТЕСТ 3: Аппроксимация trimesh (Mig29.stl)")
    print("="*60)
    
    stl_file = "Mig29.stl"
    if not os.path.exists(stl_file):
        print(f"Файл {stl_file} не найден")
        return
    
    print(f"Загрузка {stl_file}...")
    mesh_trimesh = trimesh.load(stl_file)
    vertices = mesh_trimesh.vertices
    faces = mesh_trimesh.faces
    print(f"Треугольников: {len(faces)}")
    
    print("\n1. Загрузка через trimesh:")
    start = time.time()
    mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    trimesh_load = time.time() - start
    print(f"   Время: {trimesh_load:.4f} сек")
    
    print("\n2. Построение твоего BVH:")
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
    
    start = time.time()
    mesh = Mesh(triangles)
    bvh_build = time.time() - start
    print(f"   Время: {bvh_build:.4f} сек")
    
    bounds = np.array([np.min(vertices, axis=0), np.max(vertices, axis=0)])
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
    
    depths = [1, 2, 3, 4]
    
    print(f"\n3. Рендеринг с разной глубиной:")
    print(f"{'Глубина':<10} {'Время (сек)':<15} {'Пиксели/сек':<15}")
    print("-" * 45)
    
    for depth in depths:
        renderer = MeshRenderer(
            mesh=mesh,
            camera=camera,
            plane=plane,
            light=light
        )
        
        start = time.time()
        image = renderer.render(800, 600, max_depth=depth)
        end = time.time()
        
        elapsed = end - start
        pixels = 800 * 600
        pps = pixels / elapsed
        
        print(f"{depth:<10} {elapsed:<15.2f} {pps:<15.0f}")
        
        img = Image.fromarray(image)
        img.save(f'trimesh_mig29_depth_{depth}.png')
    
    print("\n" + "="*60)
    print("ИТОГИ ТЕСТА 3:")
    print("="*60)
    print(f"• Загрузка trimesh: {trimesh_load:.4f} сек")
    print(f"• Построение BVH: {bvh_build:.4f} сек")
    print("• Рендеринг - см. таблицу выше")
    print("="*60)

if __name__ == "__main__":
    test_trimesh_approximation()