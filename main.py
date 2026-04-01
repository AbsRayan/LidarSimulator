import os
import sys
import yaml
import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout,
    QWidget, QStackedWidget, QFileDialog, QMessageBox,
    QDoubleSpinBox, QLabel, QHBoxLayout, QGroupBox, QSpacerItem, QSizePolicy
)
from gl_widget import SceneGLWidget
from config_loader import ConfigLoader


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiDAR Simulator")
        self.resize(800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)

        # Главное окно
        self.menu_page = QWidget()
        self.menu_layout = QVBoxLayout(self.menu_page)
        self.start_button = QPushButton("Открыть симуляцию")
        self.start_button.setFixedSize(250, 50)
        self.start_button.clicked.connect(self.show_simulation)
        self.menu_layout.addWidget(self.start_button)

        # Окно симуляции
        self.sim_page = QWidget()
        self.sim_layout = QHBoxLayout(self.sim_page)

        # Боковая панель управления
        self.control_panel_widget = QWidget()
        self.control_panel_widget.setFixedWidth(280)
        self.control_layout = QVBoxLayout(self.control_panel_widget)
        self.control_layout.setContentsMargins(0, 0, 0, 0)

        self.back_button = QPushButton("Назад")
        self.back_button.clicked.connect(self.show_menu)
        self.control_layout.addWidget(self.back_button)

        self.load_camera_button = QPushButton("Загрузить конфиг камеры")
        self.load_camera_button.clicked.connect(self.load_camera_config)
        self.control_layout.addWidget(self.load_camera_button)

        self.tof_button = QPushButton("📷 Снимок ToF камерой")
        self.tof_button.clicked.connect(self.take_tof_snapshot)
        self.tof_button.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold; margin-top: 10px;")
        self.control_layout.addWidget(self.tof_button)



        # Панель управления позициями объектов
        self.objects_group = QGroupBox("Управление самолётом")
        self.objects_layout = QVBoxLayout(self.objects_group)

        # Самолёт (Позиция)
        self.airplane_layout = QHBoxLayout()
        self.airplane_layout.addWidget(QLabel("Позиция:"))
        self.airplane_spins = []
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.5)
            # Начальное значение из gl_widget по умолчанию [22.0, 0.0, 0.0]
            val = 22.0 if i == 0 else 0.0
            spin.setValue(val)
            spin.valueChanged.connect(self._update_airplane_pos)
            self.airplane_spins.append(spin)
            self.airplane_layout.addWidget(spin)
        self.objects_layout.addLayout(self.airplane_layout)

        # Самолёт (Вращение)
        self.airplane_rot_layout = QHBoxLayout()
        self.airplane_rot_layout.addWidget(QLabel("Вращение:"))
        self.airplane_rot_spins = []
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-360.0, 360.0)
            spin.setSingleStep(5.0)
            # Начальное вращение из gl_widget по умолчанию [-90.0, 0.0, 0.0]
            val = -90.0 if i == 0 else 0.0
            spin.setValue(val)
            spin.valueChanged.connect(self._update_airplane_pos)
            self.airplane_rot_spins.append(spin)
            self.airplane_rot_layout.addWidget(spin)
        self.objects_layout.addLayout(self.airplane_rot_layout)

        self.control_layout.addWidget(self.objects_group)
        
        # Панель управления ToF камерой
        self.tof_ctrl_group = QGroupBox("Управление ToF камерой")
        self.tof_ctrl_layout = QVBoxLayout(self.tof_ctrl_group)
        
        self.tof_pos_layout = QHBoxLayout()
        self.tof_pos_layout.addWidget(QLabel("Позиция:"))
        self.tof_pos_spins = []
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.5)
            val = 5.0 if i == 1 else (10.0 if i == 2 else 0.0)
            spin.setValue(val)
            spin.valueChanged.connect(self._update_tof_pos)
            self.tof_pos_spins.append(spin)
            self.tof_pos_layout.addWidget(spin)
        self.tof_ctrl_layout.addLayout(self.tof_pos_layout)

        self.tof_dir_layout = QHBoxLayout()
        self.tof_dir_layout.addWidget(QLabel("Направление:"))
        self.tof_dir_spins = []
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.5)
            val = -5.0 if i == 1 else (-10.0 if i == 2 else 0.0)
            spin.setValue(val)
            spin.valueChanged.connect(self._update_tof_pos)
            self.tof_dir_spins.append(spin)
            self.tof_dir_layout.addWidget(spin)
        self.tof_ctrl_layout.addLayout(self.tof_dir_layout)
        
        self.control_layout.addWidget(self.tof_ctrl_group)

        self.control_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        self.sim_layout.addWidget(self.control_panel_widget)

        # Окно OpenGL занимает оставшееся место
        self.gl_scene = SceneGLWidget()
        self.sim_layout.addWidget(self.gl_scene, stretch=1)

        self.stacked_widget.addWidget(self.menu_page)
        self.stacked_widget.addWidget(self.sim_page)

        self._load_configs()

    def show_simulation(self):
        """Переключает интерфейс на окно симуляции"""
        self.stacked_widget.setCurrentWidget(self.sim_page)

    def show_menu(self):
        """Возвращает в главное меню"""
        self.stacked_widget.setCurrentWidget(self.menu_page)

    def _update_airplane_pos(self):
        """Обновление позиции и вращения самолёта"""
        if hasattr(self, 'gl_scene'):
            pos = [spin.value() for spin in self.airplane_spins]
            rot = [spin.value() for spin in self.airplane_rot_spins]
            self.gl_scene.airplane_pos = pos
            self.gl_scene.airplane_rot = rot
            self.gl_scene.update()

    def _update_tof_pos(self):
        """Обновление позиции и направления ToF камеры"""
        if hasattr(self, 'gl_scene'):
            pos = [spin.value() for spin in self.tof_pos_spins]
            d = [spin.value() for spin in self.tof_dir_spins]
            self.gl_scene.tof_pos = pos
            self.gl_scene.tof_dir = d
            self.gl_scene.update()



    def load_camera_config(self):
        """Открывает диалог выбора файла конфига камеры и применяет параметры к сцене"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл конфигурации камеры",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'),
            "Config files (*.yaml *.yml *.toml);;All files (*.*)"
        )
        if not path:
            return  # пользователь отменил диалог
        try:
            config = ConfigLoader.load(path)
            if config.get('type', '').lower() not in ('camera', ''):
                # Предупреждаем, но всё равно применяем — на случай конфига без поля type
                pass
            self.gl_scene.apply_camera_config(config)
            name = config.get('name', os.path.basename(path))
            QMessageBox.information(
                self,
                "Конфиг применён",
                f"Камера \«{name}\» загружена.\n"
                f"FOV: {self.gl_scene.cam_fov}°  "
                f"Near: {self.gl_scene.cam_near}  "
                f"Far: {self.gl_scene.cam_far}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))

    def take_tof_snapshot(self):
        """Запрашивает расчет ToF камеры у виджета."""
        if hasattr(self, 'gl_scene'):
            self.tof_button.setText("Расчёт... Ждите")
            self.tof_button.setEnabled(False)
            QApplication.processEvents() # Обновим UI до фриза
            
            self.gl_scene.calculate_tof()
            
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_images")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            render_path = os.path.join(output_dir, f"scene_render_{timestamp}.png")
            depth_path = os.path.join(output_dir, f"depth_map_{timestamp}.png")
            
            # Сохранение снимка сцены (рендера)
            scene_img = self.gl_scene.grabFramebuffer()
            scene_img.save(render_path)
            print(f"Рендер сцены сохранен в {render_path}")
            
            # Сохранение карты глубин
            if hasattr(self.gl_scene, 'save_depth_map'):
                self.gl_scene.save_depth_map(depth_path)
            
            self.tof_button.setText("📷 Снимок ToF камерой")
            self.tof_button.setEnabled(True)
            if getattr(self.gl_scene, 'tof_points', None) is not None:
                QMessageBox.information(self, "ToF", f"Расчёт завершен!\nПолучено точек: {len(self.gl_scene.tof_points)}\n\nФайлы сохранены в:\n{output_dir}")

    def _load_configs(self):
        """Загрузка конфигураций сенсоров"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(base_dir, 'configs', 'sensor.yaml')
        toml_path = os.path.join(base_dir, 'configs', 'sensor.toml')
        
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())