import numpy as np

import stl_reader
from tof_modeling import ToFCamera
from geometry import Triangle, Point, Sphere, Figure

from time import perf_counter


def read_stl(stl_file: str) -> Figure:
    """
    Read stl to the figure.

    Args:
        stl_file: file name of stl

    Returns: figure which consists of triangles.
    """
    vertices, index = stl_reader.read(stl_file)
    triangles = []

    counter = 0

    for ind in index:
        ind1, ind2, ind3 = ind

        v1 = Point(vertices[ind1])
        v2 = Point(vertices[ind2])
        v3 = Point(vertices[ind3])

        triangle = Triangle(v1, v2, v3)

        counter += 1

        triangles.append(triangle)

    print(f"triangles count = {counter}")

    return Figure(triangles)


def simple_pyramid(camera: ToFCamera) -> None:
    triangle_1 = Triangle(
        Point(np.array([0, 0, 2])),
        Point(np.array([2, 2, 3])),
        Point(np.array([2, -2, 3]))
    )

    triangle_2 = Triangle(
        Point(np.array([0, 0, 2])),
        Point(np.array([-2, -2, 3])),
        Point(np.array([-2, 2, 3]))
    )

    triangle_3 = Triangle(
        Point(np.array([0, 0, 2])),
        Point(np.array([2, 2, 3])),
        Point(np.array([-2, 2, 3]))
    )

    triangle_4 = Triangle(
        Point(np.array([0, 0, 2])),
        Point(np.array([-2, -2, 3])),
        Point(np.array([2, -2, 3]))
    )

    figure = Figure([triangle_1, triangle_2, triangle_3, triangle_4])

    camera.get_points_and_distances_to_object(figure)
    camera.visualize_depth_map()
    camera.visualize_point_cloud()

def simple_sphere(camera: ToFCamera, sphere: Sphere) -> None:
    camera.get_points_and_distances_to_object(sphere)
    camera.visualize_depth_map()
    camera.visualize_point_cloud()

def simple_sphere(camera: ToFCamera, triangle: Triangle) -> None:
    camera.get_points_and_distances_to_object(triangle)
    camera.visualize_depth_map()
    camera.visualize_point_cloud()

def difficult_figure(camera: ToFCamera) -> None:
    triangle_1 = Triangle(
        Point(np.array([1, 1, 1])),
        Point(np.array([1, -1, 1])),
        Point(np.array([-1, 0, 1]))
    )

    triangle_2 = Triangle(
        Point(np.array([1, 1, 1])),
        Point(np.array([1, -1, 1])),
        Point(np.array([3, 0, 3]))
    )

    triangle_3 = Triangle(
        Point(np.array([1, 1, 1])),
        Point(np.array([-1, 0, 1])),
        Point(np.array([-1, 2, 3]))
    )

    triangle_4 = Triangle(
        Point(np.array([1, -1, 1])),
        Point(np.array([-1, 0, 1])),
        Point(np.array([-1, -2, 3]))
    )
    
    triangle_5 = Triangle(
        Point(np.array([1, 1, 1])),
        Point(np.array([3, 0, 3])),
        Point(np.array([3, 3, 4]))
    )

    triangle_6 = Triangle(
        Point(np.array([1, 1, 1])),
        Point(np.array([-1, 2, 3])),
        Point(np.array([3, 3, 4]))
    )

    triangle_7 = Triangle(
        Point(np.array([1, -1, 1])),
        Point(np.array([3, 0, 3])),
        Point(np.array([3, -3, 4]))
    )

    triangle_8 = Triangle(
        Point(np.array([1, -1, 1])),
        Point(np.array([-1, -2, 3])),
        Point(np.array([3, -3, 4]))
    )

    triangles = [triangle_1, triangle_2, triangle_3, triangle_4, triangle_5, triangle_6, triangle_7, triangle_8]

    figure = Figure(triangles)

    camera.get_points_and_distances_to_object(figure)
    camera.visualize_depth_map()
    camera.visualize_point_cloud()


if __name__ == "__main__":
    start = perf_counter()

    figure = read_stl("Mig29.stl")
    figure_center = figure.get_center()

    tof_camera = ToFCamera(
        position=Point(np.array([-50, 250, 300])),
        width=100,
        height=100,
        direction=figure_center.coords - np.array([-50, 250, 300]),
        fov=60
    )

    tof_camera.get_points_and_distances_to_object_parallel(figure)
    tof_camera.visualize_depth_map()
    tof_camera.visualize_point_cloud()

    end = perf_counter()

    print(f"total time = {end - start}")
