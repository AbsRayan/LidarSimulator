import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QStackedWidget
from gl_widget import SceneGLWidget


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
        self.sim_layout = QVBoxLayout(self.sim_page)

        self.back_button = QPushButton("Назад")
        self.back_button.setFixedSize(150, 30)
        self.back_button.clicked.connect(self.show_menu)
        self.sim_layout.addWidget(self.back_button)

        self.gl_scene = SceneGLWidget()
        self.sim_layout.addWidget(self.gl_scene)

        self.stacked_widget.addWidget(self.menu_page)
        self.stacked_widget.addWidget(self.sim_page)

    def show_simulation(self):
        """Переключает интерфейс на окно симуляции"""
        self.stacked_widget.setCurrentWidget(self.sim_page)

    def show_menu(self):
        """Возвращает в главное меню"""
        self.stacked_widget.setCurrentWidget(self.menu_page)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())