import time
import numpy as np
from core.camera import Camera
from core.plane import Plane
from core.light import Light
from core.material import Material
from geometry.sphere import Sphere
from geometry.textured_sphere import TexturedSphere
from geometry.mesh import Mesh, Triangle
from renderer.sphere_renderer import SphereRenderer
from renderer.mesh_renderer import MeshRenderer
from PIL import Image
import colorsys
import os

def test_sphere_performance():
    print("\n" + "="*60)
    print("ТЕСТ 1: Сферы без текстурирования, отражения и преломления")
    print("="*60)
    
    camera = Camera(
        position=np.array([0, 2, 12]),
        look_at=np.array([0, 0, 0]),
        vector_up=np.array([0, 1, 0]),
        fov_vertical=45,
        fov_horizontal=60
    )
    
    plane = Plane(
        point=np.array([0, -1.5, 0]),
        normal=np.array([0, 1, 0]),
        color=np.array([180, 180, 180])
    )
    
    light = Light(
        position=np.array([5, 10, 5]),
        color=np.array([255, 255, 255]),
        intensity=10.0
    )
    
    spheres = []
    material = Material(
        color=np.array([255, 255, 255]),
        refractive_index=1.0,
        reflection=0.0,
        refraction=0.0,
        diffuse=0.9,
        specular=0.0,
        shininess=1.0
    )
    
    center_sphere = Sphere(
        radius=1.2,
        center=np.array([0, 0.5, 0]),
        color=np.array([255, 255, 255]),
        material=material
    )
    spheres.append(center_sphere)
    
    radius = 3.5
    for i in range(12):
        angle = 2 * np.pi * i / 12
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        
        hue = i / 12
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        color = np.array([int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)])
        
        sphere = Sphere(
            radius=0.7,
            center=np.array([x, -0.3, z]),
            color=color,
            material=material
        )
        spheres.append(sphere)
    
    depths = [1, 2, 3, 4, 5]
    
    print(f"\n{'Глубина':<10} {'Время (сек)':<15} {'Пиксели/сек':<15}")
    print("-" * 45)
    
    for depth in depths:
        renderer = SphereRenderer(
            spheres=spheres,
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
        img.save(f'sphere_perf_depth_{depth}.png')

def test_mesh_performance():
    print("\n" + "="*60)
    print("ТЕСТ 2: Mesh (икосаэдр) с BVH разной глубины")
    print("="*60)
    
    camera = Camera(
        position=np.array([3, 1.5, 2]),
        look_at=np.array([0, 0, -3]),
        vector_up=np.array([0, 1, 0]),
        fov_vertical=45,
        fov_horizontal=60
    )
    
    plane = Plane(
        point=np.array([0, -1, 0]),
        normal=np.array([0, 1, 0]),
        color=np.array([150, 150, 150])
    )
    
    light = Light(
        position=np.array([5, 7, 3]),
        color=np.array([255, 255, 220]),
        intensity=2.5
    )
    
    phi = (1 + np.sqrt(5)) / 2
    vertices = [
        np.array([-1,  phi, 0]), np.array([ 1,  phi, 0]),
        np.array([-1, -phi, 0]), np.array([ 1, -phi, 0]),
        np.array([0, -1,  phi]), np.array([0,  1,  phi]),
        np.array([0, -1, -phi]), np.array([0,  1, -phi]),
        np.array([ phi, 0, -1]), np.array([ phi, 0,  1]),
        np.array([-phi, 0, -1]), np.array([-phi, 0,  1])
    ]
    
    center = np.array([0, 0.5, -3])
    radius = 1.5
    vertices = [v / np.linalg.norm(v) * radius + center for v in vertices]
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    material = Material(
        color=np.array([255, 140, 100]),
        refractive_index=1.0,
        reflection=0.0,
        refraction=0.0,
        diffuse=0.9,
        specular=0.0
    )
    
    triangles = []
    for v0, v1, v2 in faces:
        a = vertices[v0]
        b = vertices[v1]
        c = vertices[v2]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
            triangles.append(Triangle(a, b, c, normal, np.array([255, 140, 100]), material))
    
    mesh = Mesh(triangles)
    
    depths = [1, 2, 3, 4, 5]
    
    print(f"\n{'Глубина':<10} {'Время (сек)':<15} {'Пиксели/сек':<15}")
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
        img.save(f'mesh_perf_depth_{depth}.png')

def test_compare():
    print("\n" + "="*60)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*60)
    
    test_sphere_performance()
    test_mesh_performance()
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ:")
    print("="*60)
    print("1. Сферы: O(N) где N - количество сфер")
    print("2. Mesh с BVH: O(log N) где N - количество треугольников")
    print("3. Глубина рекурсии линейно влияет на время")
    print("="*60)

if __name__ == "__main__":
    test_compare()