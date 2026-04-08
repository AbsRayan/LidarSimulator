import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List


def _parse_vector(values, default: List[float], cast_type=float):
    if not isinstance(values, list) or len(values) != len(default):
        return default.copy()
    return [cast_type(value) for value in values]

@dataclass
class SceneObject:
    id: str
    type: str  # 'mesh', 'plane', 'box'
    position: List[float]
    rotation: List[float]
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    color: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])
    texture: str = ""
    
    # Specifics
    model_path: str = ""
    dimensions: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    
    # Dynamic binding keys
    dynamic_pos: str = ""
    dynamic_rot: str = ""

    def get_triangles(self) -> List:
        """
        Генерирует список треугольников для примитивов (box, plane).
        Возвращает список кортежей: [((v1, v2, v3), normal), ...]
        Каждая вершина - np.array формы (3,)
        """
        tris = []
        if self.type == 'plane':
            sx, sy, sz = self.scale
            px, py, pz = self.position
            
            # Plane is defined in XZ plane at y = py
            v0 = np.array([px - sx, py, pz - sz])
            v1 = np.array([px - sx, py, pz + sz])
            v2 = np.array([px + sx, py, pz + sz])
            v3 = np.array([px + sx, py, pz - sz])
            
            normal = np.array([0.0, 1.0, 0.0])
            tris.append(((v0, v1, v2), normal))
            tris.append(((v0, v2, v3), normal))
            
        elif self.type == 'box':
            w, h, d = self.dimensions
            px, py, pz = self.position
            
            x0, x1 = px - w / 2, px + w / 2
            y0, y1 = py, py + h
            z0, z1 = pz - d / 2, pz + d / 2
            
            # Вспомогательная функция для добавления квада 
            def add_quad(pa, pb, pc, pd, n):
                tris.append(((pa, pb, pc), n))
                tris.append(((pa, pc, pd), n))
            
            # Top (0, 1, 0)
            add_quad(
                np.array([x0, y1, z0]), np.array([x0, y1, z1]),
                np.array([x1, y1, z1]), np.array([x1, y1, z0]),
                np.array([0.0, 1.0, 0.0])
            )
            # Bottom (0, -1, 0) - ignored in GL building but we can add anyway
            # Front (0, 0, 1)
            add_quad(
                np.array([x0, y0, z1]), np.array([x1, y0, z1]),
                np.array([x1, y1, z1]), np.array([x0, y1, z1]),
                np.array([0.0, 0.0, 1.0])
            )
            # Back (0, 0, -1)
            add_quad(
                np.array([x1, y0, z0]), np.array([x0, y0, z0]),
                np.array([x0, y1, z0]), np.array([x1, y1, z0]),
                np.array([0.0, 0.0, -1.0])
            )
            # Left (-1, 0, 0)
            add_quad(
                np.array([x0, y0, z0]), np.array([x0, y0, z1]),
                np.array([x0, y1, z1]), np.array([x0, y1, z0]),
                np.array([-1.0, 0.0, 0.0])
            )
            # Right (1, 0, 0)
            add_quad(
                np.array([x1, y0, z1]), np.array([x1, y0, z0]),
                np.array([x1, y1, z0]), np.array([x1, y1, z1]),
                np.array([1.0, 0.0, 0.0])
            )
            
        return tris


@dataclass
class ToFCameraConfig:
    position: List[float] = field(default_factory=lambda: [0.0, 5.0, 10.0])
    target: List[float] = field(default_factory=lambda: [22.0, 0.0, 0.0])
    fov: float = 45.0
    near: float = 0.1
    far: float = 100.0
    resolution: List[int] = field(default_factory=lambda: [100, 100])

@dataclass
class SceneConfig:
    objects: List[SceneObject]
    textures: dict = field(default_factory=dict)
    tof_camera: ToFCameraConfig = field(default_factory=ToFCameraConfig)

def load_scene(config_path: str) -> SceneConfig:
    if not os.path.exists(config_path):
        print(f"[SceneLoader] Warning: {config_path} not found. Returning empty scene.")
        return SceneConfig(objects=[])
        
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    objs = []
    for o_data in data.get("objects", []):
        obj = SceneObject(
            id=o_data.get("id", ""),
            type=o_data.get("type", "mesh"),
            position=list(o_data.get("position", [0.0, 0.0, 0.0])),
            rotation=list(o_data.get("rotation", [0.0, 0.0, 0.0])),
            scale=list(o_data.get("scale", [1.0, 1.0, 1.0])),
            color=list(o_data.get("color", [0.8, 0.8, 0.8])),
            texture=o_data.get("texture", ""),
            model_path=o_data.get("model_path", ""),
            dimensions=list(o_data.get("dimensions", [1.0, 1.0, 1.0])),
            dynamic_pos=o_data.get("dynamic_pos", ""),
            dynamic_rot=o_data.get("dynamic_rot", "")
        )
        objs.append(obj)
        
    textures = data.get("textures", {})
    tof_data = data.get("tof_camera", {})
    tof_camera = ToFCameraConfig(
        position=_parse_vector(tof_data.get("position"), [0.0, 5.0, 10.0]),
        target=_parse_vector(tof_data.get("target"), [22.0, 0.0, 0.0]),
        fov=float(tof_data.get("fov", 45.0)),
        near=float(tof_data.get("near", 0.1)),
        far=float(tof_data.get("far", 100.0)),
        resolution=_parse_vector(tof_data.get("resolution"), [100, 100], int)
    )

    return SceneConfig(objects=objs, textures=textures, tof_camera=tof_camera)
