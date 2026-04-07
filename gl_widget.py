"""
OpenGL-виджет для отрисовки 3D сцены.
"""
import os
import ctypes
import math

import numpy as np
import stl_loader
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GL import shaders as gl_shaders
from OpenGL.GLU import *


_SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shaders')


def _load_shader_source(filename: str) -> str:
    path = os.path.join(_SHADER_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


VERTEX_SHADER_SRC = _load_shader_source('vertex.glsl')
FRAGMENT_SHADER_SRC = _load_shader_source('fragment.glsl')
MODEL_VERTEX_SHADER_SRC = _load_shader_source('model_vertex.glsl')
MODEL_FRAGMENT_SHADER_SRC = _load_shader_source('model_fragment.glsl')

# Каждая строка: posX, posY, texU, texV
QUAD_VERTICES = np.array([
    # Первый треугольник
    -1.0,  1.0,  0.0, 1.0,
    -1.0, -1.0,  0.0, 0.0,
     1.0, -1.0,  1.0, 0.0,
    # Второй треугольник
    -1.0,  1.0,  0.0, 1.0,
     1.0, -1.0,  1.0, 0.0,
     1.0,  1.0,  1.0, 1.0,
], dtype=np.float32)


class SceneGLWidget(QOpenGLWidget):
    """Виджет для отрисовки 3D сцены """

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Положение камеры (для gluLookAt)
        self.camera_target = [0.0, 0.0, 0.0]
        self.camera_distance = 15.0
        self.camera_yaw = 45.0
        self.camera_pitch = 30.0
        self.camera_pos = [0.0, 0.0, 8.0]
        self.last_mouse_pos = None

        self._update_camera_from_angles()

        # Параметры перспективы (можно обновить через apply_camera_config)
        self.cam_fov = 45.0
        self.cam_near = 0.1
        self.cam_far = 100.0

        # Позиция объектов в сцене (устанавливается из SimulationController)
        self.airplane_pos = [0.0, 0.0, 0.0]
        self.airplane_rot = [0.0, 0.0, 0.0]  # Вращение (Yaw, Pitch, Roll)
        
        # Позиция и направление ToF-камеры (устанавливается из SimulationController)
        self.tof_pos = [0.0, 0.0, 0.0]
        self.tof_dir = [0.0, 0.0, 0.0]

        # OpenGL ресурсы (создаются в initializeGL)
        self.quadric = None
        self.airplane_list = None
        self.airplane_loaded = False

        # FBO ресурсы
        self.fbo = None
        self.fbo_texture = None
        self.fbo_depth = None
        self.fbo_width = 800
        self.fbo_height = 600

        # Точки ToF камеры
        self.tof_points = None

        # Шейдер и fullscreen quad
        self.shader_program = None
        self.model_shader_program = None
        self.quad_vao = None
        self.quad_vbo = None

        # Текстура проектора
        self.projector_texture = None
        self.texture_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'textures', 'base_texture.png')

        # Текстура земли
        self.ground_texture = None
        self.ground_texture_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'textures', 'ground.jpg')

        # Текстура взлетной полосы
        self.road_texture = None
        self.road_texture_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'textures', 'road.png')

        # Текстура площадки (field) в конце полосы
        self.field_texture = None
        self.field_texture_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'textures', 'field.jpg')

        # Текстуры зданий
        self.building1_texture = None
        self.building1_texture_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'textures', 'building1.jpg')
        self.building2_texture = None
        self.building2_texture_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'textures', 'building2.jpg')

        # Текстура самолета (фиксированная)
        self.airplane_texture = None
        self.airplane_texture_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'textures', 'MiG29.png')

        # Загрузка STL-модели
        self.airplane_mesh = stl_loader.load_stl('models/Mig29.stl')
        self.airplane_loaded = self.airplane_mesh is not None

    def _update_camera_from_angles(self):
        rad_yaw = math.radians(self.camera_yaw)
        rad_pitch = math.radians(self.camera_pitch)
        
        x = self.camera_target[0] + self.camera_distance * math.cos(rad_pitch) * math.sin(rad_yaw)
        y = self.camera_target[1] + self.camera_distance * math.sin(rad_pitch)
        z = self.camera_target[2] + self.camera_distance * math.cos(rad_pitch) * math.cos(rad_yaw)
        self.camera_pos = [x, y, z]

    def _update_angles_from_camera(self):
        dx = self.camera_pos[0] - self.camera_target[0]
        dy = self.camera_pos[1] - self.camera_target[1]
        dz = self.camera_pos[2] - self.camera_target[2]
        
        self.camera_distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if self.camera_distance > 0.0001:
            self.camera_pitch = math.degrees(math.asin(max(-1.0, min(1.0, dy / self.camera_distance))))
            self.camera_yaw = math.degrees(math.atan2(dx, dz))


    def initializeGL(self):
        self._init_scene_resources()
        self._build_airplane_display_list()
        self._create_shader()
        self._create_fullscreen_quad()
        self._create_fbo(self.width() or 800, self.height() or 600)
        if self.projector_texture is None:
            self.projector_texture = self._load_texture(self.texture_path)
        if self.ground_texture is None:
            self.ground_texture = self._load_texture(self.ground_texture_path, wrap=GL_REPEAT)
        if self.road_texture is None:
            self.road_texture = self._load_texture(self.road_texture_path, wrap=GL_REPEAT)
        if self.field_texture is None:
            self.field_texture = self._load_texture(self.field_texture_path, wrap=GL_REPEAT)
        if self.building1_texture is None:
            self.building1_texture = self._load_texture(self.building1_texture_path, wrap=GL_REPEAT)
        if self.building2_texture is None:
            self.building2_texture = self._load_texture(self.building2_texture_path, wrap=GL_REPEAT)
        if self.airplane_texture is None:
            self.airplane_texture = self._load_texture(self.airplane_texture_path, wrap=GL_REPEAT)

    def _init_scene_resources(self):
        """Инициализация ресурсов для 3D-сцены"""
        if self.quadric is not None:
            gluDeleteQuadric(self.quadric)
        self.quadric = gluNewQuadric()

        glClearColor(0.4, 0.5, 0.6, 1.0) # Сделаем фон приятным (серо-голубым вместо темно-серого)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # Общее фоновое освещение (делает тени мягче и ярче)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.7, 0.7, 0.7, 1.0])
        
        # Настраиваем сам источник света
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.6, 0.6, 0.6, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    def _build_airplane_display_list(self):
        """Создание display list для быстрой отрисовки модели самолёта"""
        if not self.airplane_loaded:
            return
        self.airplane_list = stl_loader.build_display_list(self.airplane_mesh)


    def _create_fbo(self, w, h):
        """Создание / пересоздание Frame Buffer Object"""
        # Удаляем старые ресурсы
        if self.fbo is not None:
            glDeleteFramebuffers(1, [self.fbo])
        if self.fbo_texture is not None:
            glDeleteTextures(1, [self.fbo_texture])
        if self.fbo_depth is not None:
            glDeleteRenderbuffers(1, [self.fbo_depth])

        dpr = self.devicePixelRatioF()
        self.fbo_width = max(int(w * dpr), 1)
        self.fbo_height = max(int(h * dpr), 1)

        # Цветовая текстура
        self.fbo_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.fbo_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.fbo_width, self.fbo_height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Renderbuffer для глубины
        self.fbo_depth = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.fbo_depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
                              self.fbo_width, self.fbo_height)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        # Собираем FBO
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, self.fbo_texture, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, self.fbo_depth)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Ошибка FBO! Статус: {status}")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)


    def _load_texture(self, file_path, wrap=GL_CLAMP_TO_BORDER):
        """Загружает текстуру из файла."""
        img = QImage(file_path)
        if img.isNull():
            print(f"Failed to load image: {file_path}")
            return None
            
        img = img.convertToFormat(QImage.Format.Format_RGBA8888)
        width = img.width()
        height = img.height()
        
        ptr = img.constBits()
        ptr.setsize(width * height * 4)
        data = np.frombuffer(ptr, dtype=np.uint8).copy()
        
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap)
        if wrap == GL_CLAMP_TO_BORDER:
            glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [0.0, 0.0, 0.0, 0.0])
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glBindTexture(GL_TEXTURE_2D, 0)
        
        return tex_id

    def _get_projector_matrix(self):
        """Вычисляет матрицу проектора для переданного объекта (в локальных координатах объекта)"""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        
        # Bias (переход в диапазон [0, 1] для текстурных координат)
        glTranslatef(0.5, 0.5, 0.5)
        glScalef(0.5, 0.5, 0.5)
        
        # Проекция проектора (угол обзора)
        gluPerspective(30.0, 1.0, 0.1, 100.0)
        
        # Вид проектора
        gluLookAt(2.0, 3.0, 4.0,  # позиция проектора
                  2.0, 0.0, 0.0,  # точка, куда он смотрит 
                  0.0, 1.0, 0.0)  # вектор "вверх"
        
        # Перенос из локальных координат модели в мировые, 
        # чтобы координаты вершин совпадали с пространством проектора.
        glTranslatef(self.airplane_pos[0], self.airplane_pos[1], self.airplane_pos[2])
        glRotatef(self.airplane_rot[0], 0.0, 1.0, 0.0)
        glRotatef(self.airplane_rot[1], 1.0, 0.0, 0.0)
        glRotatef(self.airplane_rot[2], 0.0, 0.0, 1.0)
            
        proj_matrix = glGetFloatv(GL_PROJECTION_MATRIX)
        
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        return proj_matrix


    def _draw_building(self, x, y, z, w=4.0, d=4.0, h=5.0, color=(0.6, 0.6, 0.6), texture=None):
        """Рисует простой прямоугольный ящик (здание) с центром основания в (x, y, z)."""
        x0, x1 = x - w / 2, x + w / 2
        y0, y1 = y,         y + h
        z0, z1 = z - d / 2, z + d / 2

        r, g, b = color

        glPushMatrix()
        glColor3f(r, g, b)

        use_tex = texture is not None
        if use_tex:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, texture)
        else:
            glDisable(GL_TEXTURE_2D)

        glBegin(GL_QUADS)
        # Передняя грань
        glNormal3f(0, 0, 1)
        if use_tex: glTexCoord2f(0, 0)
        glVertex3f(x0, y0, z1)
        if use_tex: glTexCoord2f(1, 0)
        glVertex3f(x1, y0, z1)
        if use_tex: glTexCoord2f(1, 1)
        glVertex3f(x1, y1, z1)
        if use_tex: glTexCoord2f(0, 1)
        glVertex3f(x0, y1, z1)
        # Задняя грань
        glNormal3f(0, 0, -1)
        if use_tex: glTexCoord2f(0, 0)
        glVertex3f(x1, y0, z0)
        if use_tex: glTexCoord2f(1, 0)
        glVertex3f(x0, y0, z0)
        if use_tex: glTexCoord2f(1, 1)
        glVertex3f(x0, y1, z0)
        if use_tex: glTexCoord2f(0, 1)
        glVertex3f(x1, y1, z0)
        # Левая грань
        glNormal3f(-1, 0, 0)
        if use_tex: glTexCoord2f(0, 0)
        glVertex3f(x0, y0, z0)
        if use_tex: glTexCoord2f(1, 0)
        glVertex3f(x0, y0, z1)
        if use_tex: glTexCoord2f(1, 1)
        glVertex3f(x0, y1, z1)
        if use_tex: glTexCoord2f(0, 1)
        glVertex3f(x0, y1, z0)
        # Правая грань
        glNormal3f(1, 0, 0)
        if use_tex: glTexCoord2f(0, 0)
        glVertex3f(x1, y0, z1)
        if use_tex: glTexCoord2f(1, 0)
        glVertex3f(x1, y0, z0)
        if use_tex: glTexCoord2f(1, 1)
        glVertex3f(x1, y1, z0)
        if use_tex: glTexCoord2f(0, 1)
        glVertex3f(x1, y1, z1)
        glEnd()

        # Крыша (тёмная, без текстуры)
        if use_tex:
            glDisable(GL_TEXTURE_2D)
            
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glColor3f(0.2, 0.2, 0.2)
        glVertex3f(x0, y1, z0); glVertex3f(x0, y1, z1)
        glVertex3f(x1, y1, z1); glVertex3f(x1, y1, z0)
        glEnd()

        if use_tex:
            glBindTexture(GL_TEXTURE_2D, 0)
            glDisable(GL_TEXTURE_2D)

        glPopMatrix()


    def _create_shader(self):

        """Компиляция шейдерной программы"""
        vertex = gl_shaders.compileShader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER)
        fragment = gl_shaders.compileShader(FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER)
        self.shader_program = gl_shaders.compileProgram(vertex, fragment)
        
        m_vertex = gl_shaders.compileShader(MODEL_VERTEX_SHADER_SRC, GL_VERTEX_SHADER)
        m_fragment = gl_shaders.compileShader(MODEL_FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER)
        self.model_shader_program = gl_shaders.compileProgram(m_vertex, m_fragment)


    def _create_fullscreen_quad(self):
        """Создание VAO/VBO для полноэкранного прямоугольника"""
        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)

        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, QUAD_VERTICES.nbytes, QUAD_VERTICES, GL_STATIC_DRAW)

        stride = 4 * ctypes.sizeof(ctypes.c_float)  # 4 floats per vertex
        # Позиция (location = 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # Текстурные координаты (location = 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride,
                              ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)


    def resizeGL(self, w, h):
        """Вызывается при изменении размера окна"""
        if self.fbo is not None:
            self._create_fbo(w, h)


    def paintGL(self):
        """Основной цикл отрисовки"""
        if self.fbo is None:
            return

        if self.quadric is None:
            self._init_scene_resources()

        dpr = self.devicePixelRatioF()
        w = int(self.width() * dpr)
        h = int(self.height() * dpr)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.fbo_width, self.fbo_height)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.fbo_width / max(self.fbo_height, 1)
        gluPerspective(self.cam_fov, aspect, self.cam_near, self.cam_far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        gluLookAt(
            self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
            self.camera_target[0], self.camera_target[1], self.camera_target[2],
            0.0, 1.0, 0.0
        )

        glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, -5, 1))

        # Отрисовка плоскости (пола) с текстурой до включения шейдера
        glPushMatrix()
        glTranslatef(0.0, -0.3, 0.0) # подогнали высоту пола под шасси самолета
        
        # Включаем текстурирование
        glEnable(GL_TEXTURE_2D)
        if self.ground_texture is not None:
            glBindTexture(GL_TEXTURE_2D, self.ground_texture)
            glColor3f(1.0, 1.0, 1.0) # белый цвет, чтобы не было затемнения текстуры
        else:
            glBindTexture(GL_TEXTURE_2D, 0)
            glColor3f(0.35, 0.35, 0.35)

        # Сплошной пол
        glBegin(GL_QUADS)
        glNormal3f(0.0, 1.0, 0.0)
        
        ground_size = 100.0
        tiles = ground_size / 3.0 # Сохраняем масштаб текстуры (3 единицы на тайл)
        
        glTexCoord2f(0.0, 0.0); glVertex3f(-ground_size, 0.0, -ground_size)
        glTexCoord2f(0.0, tiles); glVertex3f(-ground_size, 0.0,  ground_size)
        glTexCoord2f(tiles, tiles); glVertex3f( ground_size, 0.0,  ground_size)
        glTexCoord2f(tiles, 0.0); glVertex3f( ground_size, 0.0, -ground_size)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        
        glPopMatrix()

        # Отрисовка взлетной полосы
        glPushMatrix()
        glTranslatef(0.0, -0.29, 0.0)
        
        glEnable(GL_TEXTURE_2D)
        if self.road_texture is not None:
            glBindTexture(GL_TEXTURE_2D, self.road_texture)
            glColor3f(1.0, 1.0, 1.0)
        else:
            glBindTexture(GL_TEXTURE_2D, 0)
            glColor3f(0.2, 0.2, 0.2)

        glBegin(GL_QUADS)
        glNormal3f(0.0, 1.0, 0.0)
        
        road_width = 3.0
        road_length = 30.0
        tiles_x = road_length / 4.0
        
        glTexCoord2f(0.0, 0.0); glVertex3f(-road_length, 0.0, -road_width)
        glTexCoord2f(0.0, 1.0); glVertex3f(-road_length, 0.0,  road_width)
        glTexCoord2f(tiles_x, 1.0); glVertex3f( road_length, 0.0,  road_width)
        glTexCoord2f(tiles_x, 0.0); glVertex3f( road_length, 0.0, -road_width)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        
        glPopMatrix()

        # Отрисовка небольшой площади (field) на конце полосы
        glPushMatrix()
        glTranslatef(0.0, -0.28, 0.0) # чуть выше взлетной полосы и пола
        
        glEnable(GL_TEXTURE_2D)
        if self.field_texture is not None:
            glBindTexture(GL_TEXTURE_2D, self.field_texture)
            glColor3f(1.0, 1.0, 1.0)
        else:
            glBindTexture(GL_TEXTURE_2D, 0)
            glColor3f(0.3, 0.3, 0.3)

        glBegin(GL_QUADS)
        glNormal3f(0.0, 1.0, 0.0)
        
        field_length = 15.0
        field_width = 10.0
        
        # Площадка, сдвинутая чуть ближе к центру (например, последние 10 метров полосы)
        start_x = road_length - field_length
        end_x = road_length
        
        glTexCoord2f(0.0, 0.0); glVertex3f( start_x, 0.0, -field_width)
        glTexCoord2f(0.0, 1.0); glVertex3f( start_x, 0.0,  field_width)
        glTexCoord2f(1.0, 1.0); glVertex3f( end_x, 0.0,  field_width)
        glTexCoord2f(1.0, 0.0); glVertex3f( end_x, 0.0, -field_width)
        
        glEnd()
        glDisable(GL_TEXTURE_2D)
        
        glPopMatrix()

        # Отрисовка зданий возле площадки
        glDisable(GL_TEXTURE_2D)
        self._draw_building(20.0, -0.3,  14.0, w=4.0, d=5.0, h=3.0, color=(1.0, 1.0, 1.0), texture=self.building1_texture)
        self._draw_building(25.0, -0.3, -14.0, w=5.0, d=4.0, h=4.5, color=(1.0, 1.0, 1.0), texture=self.building2_texture)
        # Здания напротив взлетной полосы (поперек)
        self._draw_building(34.0, -0.3,   5.0, w=6.0, d=6.0, h=5.5, color=(1.0, 1.0, 1.0), texture=self.building1_texture)
        self._draw_building(34.0, -0.3,  -5.0, w=4.5, d=5.5, h=3.5, color=(1.0, 1.0, 1.0), texture=self.building2_texture)

        # Настраиваем шейдер для проекции на модели
        glUseProgram(self.model_shader_program)
        
        loc_tex = glGetUniformLocation(self.model_shader_program, "projectorTexture")
        loc_use_proj = glGetUniformLocation(self.model_shader_program, "useProjector")
        loc_mat = glGetUniformLocation(self.model_shader_program, "projectorMatrix")
        
        loc_base_tex = glGetUniformLocation(self.model_shader_program, "baseTexture")
        loc_use_base = glGetUniformLocation(self.model_shader_program, "useBaseTexture")
        
        glUniform1i(loc_tex, 1) # Текстурный юнит 1 (проектор)
        glUniform1i(loc_base_tex, 0) # Текстурный юнит 0 (базовая)
        
        # Настройка текстуры проектора (выключена для самолета)
        glActiveTexture(GL_TEXTURE1)
        if self.projector_texture is not None:
            glBindTexture(GL_TEXTURE_2D, self.projector_texture)
            glUniform1i(loc_use_proj, 0) # Отключаем, чтобы текстура привязалась к модели
        else:
            glBindTexture(GL_TEXTURE_2D, 0)
            glUniform1i(loc_use_proj, 0)
            
        # Настройка закрепленной базовой текстуры
        glActiveTexture(GL_TEXTURE0)
        if self.airplane_texture is not None:
            glBindTexture(GL_TEXTURE_2D, self.airplane_texture)
            glUniform1i(loc_use_base, 1)
        else:
            glBindTexture(GL_TEXTURE_2D, 0)
            glUniform1i(loc_use_base, 0)

        # Самолёт
        if self.airplane_list is not None:
            airplane_proj = self._get_projector_matrix()
            glUniformMatrix4fv(loc_mat, 1, GL_FALSE, airplane_proj)
            
            # Главный (управляемый) самолет
            glPushMatrix()
            glTranslatef(self.airplane_pos[0], self.airplane_pos[1], self.airplane_pos[2])
            glRotatef(self.airplane_rot[0], 0.0, 1.0, 0.0)
            glRotatef(self.airplane_rot[1], 1.0, 0.0, 0.0)
            glRotatef(self.airplane_rot[2], 0.0, 0.0, 1.0)
            glColor3f(0.8, 0.8, 0.8) # Белый цвет, чтобы оригинальная текстура не окрашивалась
            glCallList(self.airplane_list)
            glPopMatrix()

            # 5 неподвижных самолётов сбоку площадки, близко друг к другу
            for i in range(5):
                # Z сбоку площадки
                z_pos = -7.5
                # X выстраиваем в ряд вдоль края (от 18.0 до 28.0)
                x_pos = 18.0 + i * 2.5
                
                glPushMatrix()
                glTranslatef(x_pos, 0.0, z_pos)
                glRotatef(-90.0, 0.0, 1.0, 0.0) # Носом к центру площадки
                
                glColor3f(0.8, 0.8, 0.8)
                glCallList(self.airplane_list)
                glPopMatrix()

        glUseProgram(0)

        # --- Маркер ToF камеры ---
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_LIGHTING)

        # Красная сфера в позиции ToF камеры
        glPushMatrix()
        glTranslatef(self.tof_pos[0], self.tof_pos[1], self.tof_pos[2])
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, [0.8, 0.0, 0.0, 1.0])  # Само-светящаяся
        glColor3f(1.0, 0.0, 0.0)
        gluSphere(self.quadric, 0.2, 16, 16)
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, [0.0, 0.0, 0.0, 1.0])  # Сброс emission
        glPopMatrix()

        # Линия от позиции камеры к цели (направлению)
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glColor3f(1.0, 0.4, 0.0)
        glBegin(GL_LINES)
        glVertex3f(self.tof_pos[0], self.tof_pos[1], self.tof_pos[2])
        target = [self.tof_pos[i] + self.tof_dir[i] for i in range(3)]
        glVertex3f(target[0], target[1], target[2])
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

        # Отрисовка облака точек убрана по требованию

        glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        glViewport(0, 0, w, h)

        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        glUseProgram(self.shader_program)

        # Привязываем текстуру FBO
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.fbo_texture)
        loc = glGetUniformLocation(self.shader_program, "screenTexture")
        glUniform1i(loc, 0)

        # Рисуем quad
        glBindVertexArray(self.quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)


    def apply_camera_config(self, config: dict):
        """
        Применяет параметры камеры из словаря конфигурации.
        Поддерживаемые ключи: fov, near, far, resolution.
        Вызывает обновление сцены сразу после применения.
        """
        if 'fov' in config:
            self.cam_fov = float(config['fov'])
        if 'near' in config:
            self.cam_near = float(config['near'])
        if 'far' in config:
            self.cam_far = float(config['far'])
        # resolution влияет только на aspect ratio FBO — пересоздавать FBO не нужно,
        # т.к. aspect ratio берётся из реального размера виджета.
        # Но сохраняем значение для возможного будущего использования.
        if 'resolution' in config:
            res = config['resolution']
            if isinstance(res, (list, tuple)) and len(res) == 2:
                self._config_resolution = (int(res[0]), int(res[1]))
        if 'position' in config:
            pos = config['position']
            if isinstance(pos, (list, tuple)) and len(pos) == 3:
                self.camera_pos = [float(pos[0]), float(pos[1]), float(pos[2])]
        if 'target' in config:
            tgt = config['target']
            if isinstance(tgt, (list, tuple)) and len(tgt) == 3:
                self.camera_target = [float(tgt[0]), float(tgt[1]), float(tgt[2])]

        self._update_angles_from_camera()
                
        print(f"[Camera] pos={self.camera_pos} fov={self.cam_fov}° near={self.cam_near} far={self.cam_far}")
        self.update()

    def cleanup(self):
        """Очистка всех OpenGL ресурсов"""
        self.makeCurrent()
        if self.quadric is not None:
            gluDeleteQuadric(self.quadric)
            self.quadric = None
        if self.airplane_list is not None:
            glDeleteLists(self.airplane_list, 1)
            self.airplane_list = None
        if self.fbo is not None:
            glDeleteFramebuffers(1, [self.fbo])
            self.fbo = None
        if self.fbo_texture is not None:
            glDeleteTextures(1, [self.fbo_texture])
            self.fbo_texture = None
        if self.fbo_depth is not None:
            glDeleteRenderbuffers(1, [self.fbo_depth])
            self.fbo_depth = None
        if self.shader_program is not None:
            glDeleteProgram(self.shader_program)
            self.shader_program = None
        if self.model_shader_program is not None:
            glDeleteProgram(self.model_shader_program)
            self.model_shader_program = None
        if self.projector_texture is not None:
            glDeleteTextures(1, [self.projector_texture])
            self.projector_texture = None
        if self.ground_texture is not None:
            glDeleteTextures(1, [self.ground_texture])
            self.ground_texture = None
        if self.road_texture is not None:
            glDeleteTextures(1, [self.road_texture])
            self.road_texture = None
        if self.field_texture is not None:
            glDeleteTextures(1, [self.field_texture])
            self.field_texture = None
        if self.building1_texture is not None:
            glDeleteTextures(1, [self.building1_texture])
            self.building1_texture = None
        if self.building2_texture is not None:
            glDeleteTextures(1, [self.building2_texture])
            self.building2_texture = None
        if self.airplane_texture is not None:
            glDeleteTextures(1, [self.airplane_texture])
            self.airplane_texture = None
        if self.quad_vao is not None:
            glDeleteVertexArrays(1, [self.quad_vao])
            self.quad_vao = None
        if self.quad_vbo is not None:
            glDeleteBuffers(1, [self.quad_vbo])
            self.quad_vbo = None
        self.doneCurrent()

    def hideEvent(self, event):
        """Вызывается при скрытии виджета"""
        self.cleanup()
        super().hideEvent(event)

    def showEvent(self, event):
        """Пересоздаём GL-ресурсы при повторном показе виджета"""
        super().showEvent(event)
        if self.fbo is None:
            self.makeCurrent()
            self._init_scene_resources()
            self._build_airplane_display_list()
            self._create_shader()
            self._create_fullscreen_quad()
            self._create_fbo(self.width() or 800, self.height() or 600)
            if self.projector_texture is None:
                self.projector_texture = self._load_texture(self.texture_path)
            if self.ground_texture is None:
                self.ground_texture = self._load_texture(self.ground_texture_path, wrap=GL_REPEAT)
            if self.road_texture is None:
                self.road_texture = self._load_texture(self.road_texture_path, wrap=GL_REPEAT)
            if self.field_texture is None:
                self.field_texture = self._load_texture(self.field_texture_path, wrap=GL_REPEAT)
            if self.building1_texture is None:
                self.building1_texture = self._load_texture(self.building1_texture_path, wrap=GL_REPEAT)
            if self.building2_texture is None:
                self.building2_texture = self._load_texture(self.building2_texture_path, wrap=GL_REPEAT)
            if self.airplane_texture is None:
                self.airplane_texture = self._load_texture(self.airplane_texture_path, wrap=GL_REPEAT)
            self.doneCurrent()
            self.update()

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is None:
            self.last_mouse_pos = event.position()
            return

        dx = event.position().x() - self.last_mouse_pos.x()
        dy = event.position().y() - self.last_mouse_pos.y()
        self.last_mouse_pos = event.position()

        if event.buttons() & Qt.MouseButton.LeftButton:
            # Вращение камеры
            self.camera_yaw -= dx * 0.5
            self.camera_pitch += dy * 0.5
            # Ограничение pitch, чтобы избежать переворота камеры (gimbal lock)
            self.camera_pitch = max(-89.0, min(89.0, self.camera_pitch))
            self._update_camera_from_angles()
            self.update()
        elif event.buttons() & Qt.MouseButton.RightButton:
            # Панорамирование (перемещение точки интереса)
            rad_yaw = math.radians(self.camera_yaw)
            right_x = math.cos(rad_yaw)
            right_z = -math.sin(rad_yaw)
            
            pan_speed = self.camera_distance * 0.002
            
            self.camera_target[0] -= right_x * dx * pan_speed
            self.camera_target[2] -= right_z * dx * pan_speed
            self.camera_target[1] += dy * pan_speed
            
            self._update_camera_from_angles()
            self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        
        rad_yaw = math.radians(self.camera_yaw)
        rad_pitch = math.radians(self.camera_pitch)
        
        # Вектор направления взгляда (от камеры к цели)
        dir_x = -math.cos(rad_pitch) * math.sin(rad_yaw)
        dir_y = -math.sin(rad_pitch)
        dir_z = -math.cos(rad_pitch) * math.cos(rad_yaw)
        
        # Скорость перемещения
        move_speed = 1.5
        
        if delta > 0:
            self.camera_target[0] += dir_x * move_speed
            self.camera_target[1] += dir_y * move_speed
            self.camera_target[2] += dir_z * move_speed
        else:
            self.camera_target[0] -= dir_x * move_speed
            self.camera_target[1] -= dir_y * move_speed
            self.camera_target[2] -= dir_z * move_speed
            
        self._update_camera_from_angles()
        self.update()

    def calculate_tof(self):
        """Интеграция с ToF камерой Никиты. Запускает расчет и кэширует облако точек."""
        if not self.airplane_loaded:
            return
            
        try:
            import sys
            import os
            tof_path = os.path.join(os.path.dirname(__file__), 'tof_camera')
            if tof_path not in sys.path:
                sys.path.insert(0, tof_path)
                
            from geometry import Point as ToFPoint, Triangle as ToFTriangle, Figure as ToFFigure
            from tof_modeling import ToFCamera
            
            # Нормализация модели (как в stl_loader.build_display_list)
            all_points = self.airplane_mesh.vectors.reshape(-1, 3)
            min_coords = all_points.min(axis=0)
            max_coords = all_points.max(axis=0)
            center = (min_coords + max_coords) / 2.0
            size = (max_coords - min_coords).max()
            scale = 2.0 / size if size > 0 else 1.0

            # Матрица вращения самолета
            import math
            cy = math.cos(math.radians(self.airplane_rot[0]))
            sy = math.sin(math.radians(self.airplane_rot[0]))
            cp = math.cos(math.radians(self.airplane_rot[1]))
            sp = math.sin(math.radians(self.airplane_rot[1]))
            cr = math.cos(math.radians(self.airplane_rot[2]))
            sr = math.sin(math.radians(self.airplane_rot[2]))
            
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
            Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
            rot_matrix = Ry @ Rx @ Rz

            # Собираем фигуру с учетом смещения самолета, вращения и масштабирования
            offset = np.array(self.airplane_pos)
            triangles = []
            for vec in self.airplane_mesh.vectors:
                try:
                    v1 = rot_matrix @ ((vec[0] - center) * scale) + offset
                    v2 = rot_matrix @ ((vec[1] - center) * scale) + offset
                    v3 = rot_matrix @ ((vec[2] - center) * scale) + offset
                    p1 = ToFPoint(v1)
                    p2 = ToFPoint(v2)
                    p3 = ToFPoint(v3)
                    triangles.append(ToFTriangle(p1, p2, p3))
                except ValueError:
                    pass
            
            if not triangles:
                return
            figure = ToFFigure(triangles=triangles, use_octree=True)
            
            direction = np.array(self.tof_dir)
            if np.linalg.norm(direction) < 1e-5:
                direction = np.array([0.0, 0.0, -1.0])
                
            # Инициализируем камеру (разрешение 100x100 для скорости)
            cam = ToFCamera(
                position=ToFPoint(np.array(self.tof_pos)),
                width=100,
                height=100,
                direction=direction,
                fov=self.cam_fov
            )
            
            # Расчёт для самолёта
            cam.get_points_and_distances_to_object(figure, parallel=False, use_octree=True)
            self.tof_distances = cam.object_distances.copy()
            self.tof_resolution = (100, 100)
            
            # Берём уже готовые точки прямо из камеры
            self.tof_points = cam.object_points if cam.object_points is not None else np.array([])
            
            self.update()
        except ImportError as e:
            print(f"Ошибка загрузки модулей ToF: {e}")
        except Exception as e:
            print(f"Ошибка расчёта ToF: {e}")

    def calculate_raytrace(self, output_path: str, width: int = 400, height: int = 300) -> bool:
        """
        Запускает рейтрейсинговый рендер сцены с текущей позицией камеры.
        """
        try:
            import numpy as np
            import math
            from stl_parser import parse_binary_stl
            from core.camera import Camera as RayCamera
            from core.light import Light
            from core.plane import Plane
            from core.material import Material
            from renderer.mesh_renderer import MeshRenderer
            from geometry.mesh import Mesh, Triangle as RayTriangle
            from PIL import Image

            stl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'Mig29.stl')
            airplane_color = np.array([185, 185, 185], dtype=np.float64)

            print("[Raytracer] Загрузка STL...")
            raw_mesh = parse_binary_stl(stl_path, airplane_color)

            # --- Нормализация (те же параметры, что в stl_loader) ---
            all_verts = np.array([[t.v0, t.v1, t.v2] for t in raw_mesh.triangles]).reshape(-1, 3)
            min_coords = all_verts.min(axis=0)
            max_coords = all_verts.max(axis=0)
            center = (min_coords + max_coords) / 2.0
            size = (max_coords - min_coords).max()
            scale = 2.0 / size if size > 0 else 1.0

            airplane_mat = Material(
                color=airplane_color,
                diffuse=0.9,
                specular=0.3,
                shininess=50.0,
                reflection=0.1
            )

            def _make_rot(yaw, pitch, roll):
                cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
                cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
                Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
                Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
                return Ry @ Rx @ Rz

            def _add_airplane(tris, rot_m, offs):
                for tri in raw_mesh.triangles:
                    v0 = rot_m @ ((tri.v0 - center) * scale) + offs
                    v1 = rot_m @ ((tri.v1 - center) * scale) + offs
                    v2 = rot_m @ ((tri.v2 - center) * scale) + offs
                    n = rot_m @ tri.normal
                    nl = np.linalg.norm(n)
                    if nl < 1e-6:
                        continue
                    tris.append(RayTriangle(v0, v1, v2, n / nl, airplane_color, airplane_mat))

            all_triangles = []

            # --- Главный самолёт ---
            print("[Raytracer] Добавление главного самолёта...")
            rot_main = _make_rot(self.airplane_rot[0], self.airplane_rot[1], self.airplane_rot[2])
            _add_airplane(all_triangles, rot_main, np.array(self.airplane_pos, dtype=np.float64))

            # --- 5 припаркованных самолётов (как в paintGL) ---
            print("[Raytracer] Добавление 5 припаркованных самолётов...")
            rot_parked = _make_rot(-90.0, 0.0, 0.0)
            for i in range(5):
                _add_airplane(all_triangles, rot_parked, np.array([18.0 + i * 2.5, 0.0, -7.5]))

            # --- Пол как два треугольника ---
            ground_color = np.array([70, 100, 60], dtype=np.float64)
            ground_mat = Material(color=ground_color, diffuse=0.85)
            gn = np.array([0.0, 1.0, 0.0])
            gs, gy = 100.0, -0.3
            all_triangles.append(RayTriangle(np.array([-gs,gy,-gs]), np.array([gs,gy,-gs]), np.array([gs,gy,gs]),  gn, ground_color, ground_mat))
            all_triangles.append(RayTriangle(np.array([-gs,gy,-gs]), np.array([gs,gy, gs]), np.array([-gs,gy,gs]), gn, ground_color, ground_mat))

            if not all_triangles:
                print("[Raytracer] Нет треугольников.")
                return False

            print(f"[Raytracer] Строим BVH для {len(all_triangles)} треугольников...")
            scene_mesh = Mesh(all_triangles)

            # --- Камера из текущего состояния OpenGL-вида ---
            cam_pos = np.array(self.camera_pos, dtype=np.float64)
            cam_target = np.array(self.camera_target, dtype=np.float64)

            camera = RayCamera(
                position=cam_pos,
                look_at=cam_target,
                vector_up=np.array([0.0, 1.0, 0.0]),
                fov_vertical=self.cam_fov,
                fov_horizontal=self.cam_fov * (width / height)
            )

            # --- Источник света (над сценой) ---
            light = Light(
                position=np.array([10.0, 20.0, 10.0]),
                color=np.array([255, 255, 220], dtype=np.uint8),
                intensity=2.0
            )

            # --- Пол ---
            ground_material = Material(color=np.array([80, 100, 80], dtype=np.float64), diffuse=0.8)
            plane = Plane(
                point=np.array([0.0, -0.3, 0.0]),
                normal=np.array([0.0, 1.0, 0.0]),
                color=np.array([80, 100, 80], dtype=np.float64)
            )

            renderer = MeshRenderer(
                mesh=scene_mesh,
                camera=camera,
                plane=plane,
                light=light
            )

            print(f"[Raytracer] Рендеринг {width}x{height}...")
            image_array = renderer.render(width, height)

            img = Image.fromarray(image_array.astype(np.uint8))
            img.save(output_path)
            print(f"[Raytracer] Сохранено в {output_path}")
            return True

        except Exception as e:
            import traceback
            print(f"[Raytracer] Ошибка: {e}")
            traceback.print_exc()
            return False

    def save_depth_map(self, filename="depth_map.png"):
        """
        Сохраняет карту глубин — точная копия visualize_depth_map из tof_modeling.py,
        plt.show() заменён на plt.savefig().
        """
        if not hasattr(self, 'tof_distances') or self.tof_distances is None:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        width, height = self.tof_resolution

        depth_map = self.tof_distances
        depth_map = depth_map.reshape((width, height))

        figure, axis = plt.subplots(figsize=(8, 6))

        if np.all(np.isnan(depth_map)):
            masked_depth = np.ma.masked_where(np.ones_like(depth_map, dtype=bool), depth_map)
            axis.imshow(masked_depth, cmap='plasma_r')
        else:
            axis.imshow(depth_map, cmap='plasma_r', vmin=np.nanmin(depth_map), vmax=np.nanmax(depth_map))

        axis.axis("image")
        axis.grid(False)
        axis.axis("off")

        plt.savefig(filename)
        plt.close(figure)
        print(f"Карта глубин сохранена в {filename}")


