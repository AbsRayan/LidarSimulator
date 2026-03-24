import numpy as np
from core.camera import Camera
from core.plane import Plane
from core.light import Light
from core.material import Material
from geometry.sphere import Sphere
from geometry.textured_sphere import TexturedSphere
from renderer.sphere_renderer import SphereRenderer
from PIL import Image
import colorsys
import os

def test_reflection():
    print("Тест отражения: 12 сфер по кругу")
    
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
    
    tex_material = Material(
        color=np.array([255, 255, 255]),
        reflection=0.7,
        diffuse=0.3,
        specular=0.4,
        shininess=40
    )
    
    tex_sphere = Sphere(
        radius=1.2,
        center=np.array([0, 0.5, 0]),
        color=np.array([255, 255, 255]),
        material=tex_material
    )
    
    texture_path = 'earth.jpg'
    if os.path.exists(texture_path):
        texture_img = Image.open(texture_path)
        texture_data = np.array(texture_img)
        center_sphere = TexturedSphere(
            sphere=tex_sphere,
            texture=texture_data
        )
        print("Текстура загружена")
    else:
        center_sphere = tex_sphere
        print("Текстура не найдена, используется цвет")
    
    spheres.append(center_sphere)
    
    radius = 3.5
    for i in range(12):
        angle = 2 * np.pi * i / 12
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        
        hue = i / 12
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        color = np.array([int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)])
        
        material = Material(
            color=color,
            reflection=0.5,
            diffuse=0.5,
            specular=0.3,
            shininess=32
        )
        
        sphere = Sphere(
            radius=0.7,
            center=np.array([x, -0.3, z]),
            color=color,
            material=material
        )
        spheres.append(sphere)
    
    depths = [1, 2, 3, 4]
    for depth in depths:
        print(f"Рендер с глубиной {depth}...")
        
        renderer = SphereRenderer(
            spheres=spheres,
            camera=camera,
            plane=plane,
            light=light
        )
        
        image = renderer.render(700, 500, max_depth=depth)
        img = Image.fromarray(image)
        filename = f'reflection_test_depth{depth}.png'
        img.save(filename)
        print(f"Сохранено {filename}")

def test_refraction():
    print("\nТест преломления: прозрачный шар + земля")
    
    camera = Camera(
        position=np.array([0, 1, 8]),
        look_at=np.array([0, 1, -10]),
        vector_up=np.array([0, 1, 0]),
        fov_vertical=45,
        fov_horizontal=60
    )
    
    plane = Plane(
        point=np.array([0, -2, 0]),
        normal=np.array([0, 1, 0]),
        color=np.array([150, 150, 150])
    )
    
    light = Light(
        position=np.array([5, 8, 5]),
        color=np.array([255, 255, 255]),
        intensity=12.0
    )
    
    spheres = []
    
    earth_material = Material(
        color=np.array([255, 255, 255]),
        diffuse=0.9
    )
    
    earth_sphere = Sphere(
        radius=2.5,
        center=np.array([0, 1, -15]),
        color=np.array([255, 255, 255]),
        material=earth_material
    )
    
    texture_path = 'earth.jpg'
    if os.path.exists(texture_path):
        texture_img = Image.open(texture_path)
        texture_data = np.array(texture_img)
        earth_sphere = TexturedSphere(
            sphere=earth_sphere,
            texture=texture_data
        )
        print("Текстура земли загружена")
    
    spheres.append(earth_sphere)
    
    glass_material = Material(
        color=np.array([255, 100, 100]),
        refractive_index=1.5,
        refraction=0.8,
        reflection=0.2,
        diffuse=0.2,
        specular=0.4
    )
    
    glass_sphere = Sphere(
        radius=1.2,
        center=np.array([0, 0.5, -5]),
        color=np.array([200, 220, 255]),
        material=glass_material
    )
    spheres.append(glass_sphere)
    
    renderer = SphereRenderer(
        spheres=spheres,
        camera=camera,
        plane=plane,
        light=light
    )
    
    image = renderer.render(800, 600, max_depth=5)
    img = Image.fromarray(image)
    img.save('refraction_one_sphere.png')
    print("Сохранено refraction_one_sphere.png")

if __name__ == "__main__":
    print("="*50)
    print("ТЕСТИРОВАНИЕ ОТРАЖЕНИЯ И ПРЕЛОМЛЕНИЯ")
    print("="*50)
    
    test_reflection()
    test_refraction()
    
    print("\nВсе тесты завершены!")