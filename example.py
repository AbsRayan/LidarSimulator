"""How to use"""

from tests import read_stl
from typing import Any
from geometry import Point
from tof_modeling import ToFCamera

import numpy as np

#Example
def get_distances_and_points(path_stl: str, args: Any) -> Any:
    figure = read_stl(path_stl, use_octree=True)

    #your params of camera
    tof_camera = ToFCamera(
        position=Point(np.array([])),
        width=100,
        height=100,
        direction=np.array([]),
        fov=60
    )

    tof_camera.get_points_and_distances_to_object(figure, use_octree=True)

    distances, points = tof_camera.distances_and_points
    # YOUR CODE
    # ...
