import numpy as np
import matplotlib.pyplot as plt
import laspy

from geometry import Point, Sphere, Triangle, Figure, distance_to
from pypcd4 import pypcd4

import tof_function_parallel as tfp


class Ray:
    """
    Class for modeling rays of ToF camera. 
    """

    def __init__(self, direction: np.ndarray, start: Point) -> None:
        self.direction = direction
        self.start = start

    def sphere_intersect(self, sphere: Sphere) -> Point | None:
        """
        Method for finding the nearest point of 
        intersection with a sphere.

        Args: 
            sphere: sphere for intersection with ray.

        Returns: the nearest point of intersection in 3D.
        """
        delta_x = self.start.coords - sphere.center.coords
        A = np.dot(self.direction, self.direction)
        B = 2 * np.dot(self.direction, delta_x)
        C = np.dot(delta_x, delta_x) - (sphere.R**2)
        D = B**2 - 4 * A * C
        if (D < 0):
            return None
        else:
            t1 = (-B + np.sqrt(D)) / (2 * A)
            t2 = (-B - np.sqrt(D)) / (2 * A)

            roots = []

            if t1 >= 0:
                roots.append(t1)
            if t2 >= 0:
                roots.append(t2)

            if not roots:
                return None
            t_min = min(roots)

            return Point(self.start.coords + t_min * self.direction)
        
    def triangle_intersect(self, triangle: Triangle) -> Point | None:
        """
        Method for finding the nearest point of 
        intersection with a triangle.

        Args: 
            triangle: triangle for intersection with ray.

        Returns: the nearest point of intersection in 3D.
        """
        D = triangle.D

        a = np.dot(self.direction, triangle.normal)
        b = -D - np.dot(self.start.coords, triangle.normal)
        
        if np.allclose(a, 0):
            if np.allclose(b, 0):
                if triangle.check_point_in_triangle(self.start):
                    return self.start

                vertices = triangle.vertices
                intersections = []

                for i in range(3):
                    p1 = vertices[i].coords
                    p2 = vertices[(i + 1) % 3].coords

                    p = self.get_triangle_edge_intersect(p1, p2)

                    if p is not None:
                        dist = distance_to(self.start, p)
                        intersections.append((dist, p))

                if not intersections:
                    return None
                
                intersections.sort(key=lambda x: x[0])
                return intersections[0][1]

            else:
                return None

        t = b / a

        if t < 0:
            return None

        point = Point(self.start.coords + t * self.direction)
        if (triangle.check_point_in_triangle(point)):
            return point
        return None
    
    def get_triangle_edge_intersect(self, p1: Point, p2: Point) -> Point | None:
        """
        Method for getting intersect of ray and triangle edge.

        Args:
            p1, p2: points of triangle edge.

        Returns: point if intersect exists, else - None.
        """
        vec = p1.coords - p2.coords

        A = np.column_stack((self.direction, -vec))
        b = p1.coords - self.start.coords

        x, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)

        if rank < 2:
            return None
        
        t, u = x

        if t >= -1e-10 and 0 <= u <= 1:
            if residuals.size > 0 and residuals[0] < 1e-8:
                return Point(self.start.coords + t * self.direction)
            
        return None
    
    def get_nearest_point_of_figure(self, figure: Figure) -> Point | None:
        """
        Method for getting nearest point of figure.

        Args:
            figure: figure which consists of triangles.

        Returns: The nearest point of figure.
        """
        nearest_point = None
        nearest_dist = np.inf

        for triangle in figure.triangles:
            point = self.triangle_intersect(triangle)
            if point is not None:
                dist = distance_to(point, self.start)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_point = point

        return nearest_point


class ToFCamera:
    """
    Class for modeling ToF camera.
    """

    def __init__(
        self,
        position: Point,
        width: int,
        height: int,
        direction: np.ndarray,
        fov: float        # in degrees
    ) -> None:
        self.fov = np.deg2rad(fov)
        self.position = position
        self.direction = direction
        self.width = width
        self.height = height
        self.object_points = None
        self.object_distances = None

    def generate_rays(self) -> list[Ray]:
        """
        Method for generating rays of ToF camera.

        Returns:
            list of rays.
        """
        rays = []
        d = np.linalg.norm(self.direction)
        e1 = self.direction / d

        a = np.array([1, 0, 0])
        if (np.all((a - e1) == 0)):
            a = np.array([0, 1, 0])

        e2 = np.cross(e1, a)
        e2 /= np.linalg.norm(e2)

        e3 = np.cross(e1, e2)
        e3 /= np.linalg.norm(e3)
        
        center = self.position.coords + e1

        ratio = self.height / self.width
        half_width = np.tan(self.fov / 2)
        half_height = ratio * half_width

        abscissa = np.linspace(-half_width, half_width, self.width)
        ordinate = np.linspace(-half_height, half_height, self.height)

        for y in ordinate:
            for x in abscissa:
                pixel_point = center + x * e2 + y * e3

                ray_direction = pixel_point - self.position.coords
                ray_direction /= np.linalg.norm(ray_direction)

                ray = Ray(
                    direction=ray_direction,
                    start=self.position
                )

                rays.append(ray)

        return rays
    
    def get_points_and_distances_to_object(self, geo_object: Sphere | Triangle | Figure) -> None:
        """
        Method for getting points of the geo object and distances to the geo object.

        Args:
            geo_object: object for which distances are calculated.

        """
        distances = []
        points = []
        for ray in self.generate_rays():
            if type(geo_object) == Sphere:
                point = ray.sphere_intersect(geo_object)
            elif type(geo_object) == Triangle:
                point = ray.triangle_intersect(geo_object)
            else:
                point = ray.get_nearest_point_of_figure(geo_object)

            if point is None:
                distances.append(np.nan)
            else:
                dist = distance_to(point, self.position)
                distances.append(dist)
                points.append(point.coords)

        result_distances = np.array(distances)
        result_points = np.array(points) if points else np.array([])

        self.object_points = result_points 
        self.object_distances = result_distances
    
    def get_time(self) -> np.ndarray:
        """
        Method for getting time spent by light.

        Returns: Array of times spent by light.

        """
        c = 299792458   # speed of the light
        if self.object_distances is None:
            raise ValueError("Distances were not calculated")
        times = 2 * self.object_distances / c
        return times
        

    def visualize_depth_map(self) -> None:
        """
        Method for visualizing depth map.
        """
        if self.object_distances is None:
            raise ValueError("Distances were not calculated")

        depth_map = self.object_distances
        depth_map = depth_map.reshape((self.width, self.height))

        figure, axis = plt.subplots(figsize=(8, 6))
        axis: plt.Axes
        
        if np.all(np.isnan(depth_map)):
            masked_depth = np.ma.masked_where(np.ones_like(depth_map, dtype=bool), depth_map)
            map = axis.imshow(masked_depth, cmap='plasma_r')
        else:
            map = axis.imshow(depth_map, cmap='plasma_r', vmin=np.nanmin(depth_map), vmax=np.nanmax(depth_map))

        axis.set_title('ToF camera depth map')
        axis.axis("image")
        axis.grid(False)
        figure.colorbar(map, ax=axis, label='Distance to camera')

        plt.show()

    def visualize_point_cloud(self) -> None:
        """
        Method for visualizing point cloud.
        """
        if self.object_points is None:
            raise ValueError("Points were not calculated")

        points = self.object_points

        figure = plt.figure(figsize=(8, 6))
        axis: plt.Axes = figure.add_subplot(projection='3d')

        if points.size == 0:
            axis.scatter3D(
                self.position.coords[0],
                self.position.coords[1],
                self.position.coords[2],
                color='red', marker='^', s=100, label='Camera position'
            )
            plt.legend()
        else:
            axis.scatter3D(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c='royalblue'
            )

        axis.set_title("ToF Point Cloud")
        axis.set_xlabel("X", fontsize=10)
        axis.set_ylabel("Y", fontsize=10)
        axis.set_zlabel("Z", fontsize=10)

        plt.show()

    def generate_rays_parallel(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Parallel version of generate rays.
        """
        rays_start, rays_direction = tfp.numba_generate_rays(
            self.position.coords,
            self.direction,
            self.width,
            self.height,
            self.fov
        )
        return rays_start, rays_direction

    def get_points_and_distances_to_object_parallel(self, geo_object) -> None:
        """
        Parallel version of get points and distances to object.
        """
        rays_start, rays_direction = self.generate_rays_parallel()
        
        if isinstance(geo_object, Sphere):
            object_type = 0
            object_params = np.zeros(4)
            object_params[:3] = geo_object.center.coords
            object_params[3] = geo_object.R
            
        elif isinstance(geo_object, Triangle):
            object_type = 1
            object_params = np.zeros(13)
            object_params[0:3] = geo_object.vertices[0].coords
            object_params[3:6] = geo_object.vertices[1].coords
            object_params[6:9] = geo_object.vertices[2].coords
            object_params[9:12] = geo_object.normal
            object_params[12] = geo_object.D
            
        elif isinstance(geo_object, Figure):
            object_type = 2
            n_triangles = len(geo_object.triangles)
            object_params = np.zeros(n_triangles * 13)
            
            for i, triangle in enumerate(geo_object.triangles):
                idx = i * 13
                object_params[idx:idx+3] = triangle.vertices[0].coords
                object_params[idx+3:idx+6] = triangle.vertices[1].coords
                object_params[idx+6:idx+9] = triangle.vertices[2].coords
                object_params[idx+9:idx+12] = triangle.normal
                object_params[idx+12] = triangle.D
        
        distances, points = tfp.numba_process_all_rays(
            rays_start, rays_direction,
            object_type, object_params
        )
        
        self.object_distances = distances
        
        valid_mask = ~np.isnan(points[:, 0])
        self.object_points = points[valid_mask]

    def write_pcd(self) -> None:
        """
        Write point cloud in pcd format.
        """
        if self.object_points is None:
            raise ValueError("Points were not calculated")
        pc = pypcd4.PointCloud.from_xyz_points(self.object_points)
        pc.save("point_cloud.pcd")

    def write_las(self) -> None:
        """
        Write point cloud in las format.
        """
        if self.object_points is None:
            raise ValueError("Points were not calculated")
        
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.0001, 0.0001, 0.0001])

        las = laspy.LasData(header)

        las.x = self.object_points[:, 0]
        las.y = self.object_points[:, 1]
        las.z = self.object_points[:, 2]

        las.write("point_cloud.las")

if __name__ == "__main__":
    tof_camera = ToFCamera(
        position=Point(np.array([0.0, 0.0, 0.0])),
        width=100,
        height=100,
        direction=np.array([0.0, 0.0, 1.0]),
        fov=60
    )

    sphere = Sphere(
        R=3,
        center=Point(np.array([0, 0, 6]))
    )

    triangle = Triangle(
        Point(np.array([-0.5, -0.5, 1])),
        Point(np.array([0.5, -0.5, 1])),
        Point(np.array([0, 0.5, 2]))
    )

    tof_camera.get_points_and_distances_to_object_parallel(triangle)
    tof_camera.visualize_depth_map()
    tof_camera.visualize_point_cloud()
