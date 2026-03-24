import numpy as np
from core.camera import Camera
from core.plane import Plane
from core.light import Light
from core.material import Material
from geometry.mesh import Mesh, Triangle
from renderer.mesh_renderer import MeshRenderer
from PIL import Image

def create_icosahedron(center: np.ndarray, radius: float, color: np.ndarray) -> Mesh:
    phi = (1 + np.sqrt(5)) / 2
    
    vertices = [
        np.array([-1,  phi, 0]), np.array([ 1,  phi, 0]),
        np.array([-1, -phi, 0]), np.array([ 1, -phi, 0]),
        np.array([0, -1,  phi]), np.array([0,  1,  phi]),
        np.array([0, -1, -phi]), np.array([0,  1, -phi]),
        np.array([ phi, 0, -1]), np.array([ phi, 0,  1]),
        np.array([-phi, 0, -1]), np.array([-phi, 0,  1])
    ]
    
    vertices = [v / np.linalg.norm(v) * radius + center for v in vertices]
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    material = Material(color=color)
    triangles = []
    
    for v0, v1, v2 in faces:
        a = vertices[v0]
        b = vertices[v1]
        c = vertices[v2]
        
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
            triangles.append(Triangle(a, b, c, normal, color, material))
    
    return Mesh(triangles)

def test_icosahedron() -> None:
    camera = Camera(
        position=np.array([3, 1.5, 2]),
        look_at=np.array([0, 0, -3]),
        vector_up=np.array([0, 1, 0]),
        fov_vertical=45,
        fov_horizontal=60
    )
    
    mesh = create_icosahedron(
        center=np.array([0, 0.5, -3]),
        radius=1.5,
        color=np.array([255, 140, 100])
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
    
    renderer = MeshRenderer(
        mesh=mesh,
        camera=camera,
        plane=plane,
        light=light
    )
    
    image = renderer.render(400, 300)
    img = Image.fromarray(image)
    img.save('icosahedron.png')
    print("icosahedron.png")

if __name__ == "__main__":
    test_icosahedron()