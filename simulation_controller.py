import os
import datetime
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication
from config_loader import ConfigLoader
from tof_service import ToFService
from raytrace_service import RaytraceService
from scene_loader import load_scene

class SimulationDefaults:
    AIRPLANE_POS = [22.0, 0.0, 0.0]
    AIRPLANE_ROT = [-90.0, 0.0, 0.0]
    TOF_POS = [0.0, 5.0, 10.0]
    TOF_TARGET = [22.0, 0.0, 0.0]

class SimulationController:
    def __init__(self, view):
        self.view = view
        self.gl_scene = self.view.gl_scene
        
        self.tof_service = ToFService()
        self.raytrace_service = RaytraceService()

        self.gl_scene.airplane_pos = SimulationDefaults.AIRPLANE_POS.copy()
        self.gl_scene.airplane_rot = SimulationDefaults.AIRPLANE_ROT.copy()
        self.gl_scene.tof_pos = SimulationDefaults.TOF_POS.copy()
        self.gl_scene.tof_dir = [
            SimulationDefaults.TOF_TARGET[i] - SimulationDefaults.TOF_POS[i]
            for i in range(3)
        ]

        self._connect_signals()
        self._load_configs()

    @staticmethod
    def _set_spin_values(spins, values):
        for spin, value in zip(spins, values):
            signals_blocked = spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(signals_blocked)

    @staticmethod
    def _build_output_path(category: str, filename_prefix: str, extension: str = "png") -> str:
        base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_images")
        category_dir = os.path.join(base_output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%H-%M-%S")
        return os.path.join(category_dir, f"{filename_prefix}_{timestamp}.{extension}")

    def _apply_tof_camera(self, position, target, update_view: bool = True):
        direction = [target[i] - position[i] for i in range(3)]
        self.gl_scene.tof_pos = position.copy()
        self.gl_scene.tof_dir = direction

        if update_view:
            self._set_spin_values(self.view.tof_pos_spins, position)
            self._set_spin_values(self.view.tof_target_spins, target)

    def _connect_signals(self):
        for spin in self.view.airplane_spins:
            spin.valueChanged.connect(self.update_airplane_pos)
        for spin in self.view.airplane_rot_spins:
            spin.valueChanged.connect(self.update_airplane_pos)
        for spin in self.view.tof_pos_spins:
            spin.valueChanged.connect(self.update_tof_pos)
        for spin in self.view.tof_target_spins:
            spin.valueChanged.connect(self.update_tof_pos)
            
        self.view.load_camera_button.clicked.connect(self.load_camera_config)
        self.view.tof_button.clicked.connect(self.take_tof_snapshot)
        self.view.raytrace_button.clicked.connect(self.take_raytrace_render)

    def update_airplane_pos(self):
        pos = [spin.value() for spin in self.view.airplane_spins]
        rot = [spin.value() for spin in self.view.airplane_rot_spins]
        self.gl_scene.airplane_pos = pos
        self.gl_scene.airplane_rot = rot
        self.gl_scene.update()

    def update_tof_pos(self):
        pos = [spin.value() for spin in self.view.tof_pos_spins]
        target = [spin.value() for spin in self.view.tof_target_spins]
        self._apply_tof_camera(pos, target, update_view=False)

        scene_config = getattr(self.gl_scene.scene_state, 'scene_config', None)
        if scene_config is not None and scene_config.tof_camera is not None:
            scene_config.tof_camera.position = pos.copy()
            scene_config.tof_camera.target = target.copy()

        self.gl_scene.update()

    def load_camera_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self.view,
            "Выберите файл конфигурации камеры",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'),
            "Config files (*.yaml *.yml *.toml);;All files (*.*)"
        )
        if not path:
            return
        try:
            config = ConfigLoader.load(path)
            if config.get('type', '').lower() not in ('camera', ''):
                pass
            self.gl_scene.apply_camera_config(config)
            name = config.get('name', os.path.basename(path))
            QMessageBox.information(
                self.view,
                "Конфиг применён",
                f"Камера «{name}» загружена.\n"
                f"FOV: {self.gl_scene.cam_fov}°  "
                f"Near: {self.gl_scene.cam_near}  "
                f"Far: {self.gl_scene.cam_far}"
            )
        except Exception as e:
            QMessageBox.critical(self.view, "Ошибка загрузки", str(e))

    def take_tof_snapshot(self):
        self.view.tof_button.setText("Расчёт... Ждите")
        self.view.tof_button.setEnabled(False)
        QApplication.processEvents()

        self.tof_service.calculate_tof(
            self.gl_scene.scene_state
        )
        self.gl_scene.update()

        render_path = self._build_output_path("scene_renders", "scene_render")
        depth_path = self._build_output_path("heatmaps", "depth_map")
        point_cloud_path = self._build_output_path("point_clouds", "point_cloud", extension="pcd")

        scene_img = self.gl_scene.grabFramebuffer()
        scene_img.save(render_path)
        print(f"Рендер сцены сохранен в {render_path}")

        self.tof_service.save_depth_map(self.gl_scene.scene_state, depth_path)
        self.tof_service.save_point_cloud_pcd(point_cloud_path)

        self.view.tof_button.setText("📷 Снимок ToF камерой")
        self.view.tof_button.setEnabled(True)

    def take_raytrace_render(self):
        self.view.raytrace_button.setText("Рендеринг... Ждите")
        self.view.raytrace_button.setEnabled(False)
        QApplication.processEvents()

        render_path = self._build_output_path("raytraces", "raytrace_render")

        img = self.raytrace_service.calculate_raytrace(
            self.gl_scene.scene_state, 
            self.gl_scene.camera_controller, 
            width=400, height=300
        )

        self.view.raytrace_button.setText("🎨 Рендер (Raytracer)")
        self.view.raytrace_button.setEnabled(True)

        if img is not None:
            img.save(render_path)
            QMessageBox.information(
                self.view, "Raytracer",
                f"Рендер завершён!\nСохранён в:\n{render_path}"
            )
        else:
            QMessageBox.warning(self.view, "Raytracer", "Рендер не удался. Проверьте консоль.")

    def _load_configs(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(base_dir, 'configs', 'sensor.yaml')
        toml_path = os.path.join(base_dir, 'configs', 'sensor.toml')
        scene_path = os.path.join(base_dir, 'configs', 'scene.json')
        
        try:
            lidar_config = ConfigLoader.load(yaml_path)
            print(f"Loaded YAML config (LiDAR): {lidar_config}")
        except Exception as e:
            print(f"Error loading YAML config: {e}")
            
        try:
            camera_config = ConfigLoader.load(toml_path)
            print(f"Loaded TOML config (Camera): {camera_config}")
        except Exception as e:
            print(f"Error loading TOML config: {e}")
            
        try:
            self.gl_scene.scene_state.scene_config = load_scene(scene_path)
            print(f"Loaded JSON config (Scene) with {len(self.gl_scene.scene_state.scene_config.objects)} objects.")
            tof_camera = self.gl_scene.scene_state.scene_config.tof_camera
            self.gl_scene.scene_state.tof_resolution = tuple(tof_camera.resolution)
            self._apply_tof_camera(tof_camera.position, tof_camera.target)
            self.gl_scene.update()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading JSON scene config: {e}")
