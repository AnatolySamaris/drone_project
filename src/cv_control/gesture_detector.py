import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from geometry_msgs.msg import Twist
from cv_control.utils import ValueFilter, PlotWatcher, load_keras_model
import math

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-

class HandGestureDetector(Node):
    def __init__(self):
        super().__init__("hand_gesture_detector")

        self.simulation = True
        self.get_logger().info(f"SIMULATION MODE: << {'on' if self.simulation else 'off'} >>")

        # Внутренние параметры камеры
        self.camera_fx = None
        self.camera_fy = None
        self.camera_cx = None
        self.camera_cy = None

        self.sign_names = {
            1: 'one', 2: 'two', 3: 'three', 4: 'four', 
            5: 'five', 6: 'ok', 7: 'rock', 8: 'thumbs_up'
        }

        self.min_angle_degrees = 5 # Минимальное значение эйлерова угла
        self.max_angle_degrees = 30 # Максимальное значение эйлерова угла
        self.min_palm_height = 15   # Минимальное расстояние от камеры в сантиметрах
        self.max_palm_height = 60   # Максимальное расстояние от камеры в сантиметрах
        self.speed = 1
        
        # Загрузка модели и скалера
        # self.model, self.scaler = load_keras_model()
        self.model = load_keras_model()
        self.get_logger().info("MODEL CREATED: " + str(type(self.model)))
        
        # Подписки
        self.sub_camera_info = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10
        )
        self.sub_color = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.color_callback, 10
        )
        self.sub_depth = self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw", self.depth_callback, 10
        )
        self.get_logger().info("SUBSCRIPTIONS CREATED")

        # Паблишеры
        self.cmd_pub = self.create_publisher(Twist, "/x500/cmd_vel", 10)
        self.get_logger().info("PUBLISHERS CREATED")
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        self.cv_bridge = CvBridge()
        self.get_logger().info("HAND DETECTOR CREATED")

        # Для синхронизации
        self.last_color_frame = None
        self.last_depth_frame = None

        # Визуализация управления
        self.show_camera_processing = True
        self.control_panel_size = 600
        self.control_panel_height = self.control_panel_size
        self.control_panel_width = int(1.8 * self.control_panel_size)
        self.control_panel = None
        self.update_control_panel(0, 0, 0, 0)
        self.get_logger().info("CONTROL PANEL CREATED")

        # Для фильтрации данных
        filter_window_size = 5
        self.throttle_filter = ValueFilter(filter_window_size, 0.5)
        self.roll_filter = ValueFilter(filter_window_size, 0.5)
        self.pitch_filter = ValueFilter(filter_window_size, 0.5)
        self.yaw_filter = ValueFilter(filter_window_size, 0.5)
        self.get_logger().info("CONTROL FILTERS CREATED")

        # Для мониторинга данных
        self.control_monitoring_on = False  # Включает/Выключает графики управления
        self.control_plot_window_size = 100
        self.throttle_watcher = PlotWatcher(
            ["Raw data", "Filtered data"],
            self.control_plot_window_size, figsize=(8, 4),
            plot_title="Throttle watcher"
        )
        self.roll_watcher = PlotWatcher(
            ["Raw data", "Filtered data"],
            self.control_plot_window_size, figsize=(8, 4),
            plot_title="Roll watcher"
        )
        self.pitch_watcher = PlotWatcher(
            ["Raw data", "Filtered data"],
            self.control_plot_window_size, figsize=(8, 4),
            plot_title="Pitch watcher"
        )
        self.yaw_watcher = PlotWatcher(
            ["Raw data", "Filtered data"],
            self.control_plot_window_size, figsize=(8, 4),
            plot_title="Yaw watcher"
        )
        self.throttle_plot = None
        self.roll_plot = None
        self.pitch_plot = None
        self.yaw_plot = None
        self.get_logger().info("DATA WATCHERS CREATED")

    def camera_info_callback(self, msg):
        K_matrix = np.array(msg.k).reshape(3, 3)
        self.camera_fx = K_matrix[0, 0]
        self.camera_fy = K_matrix[1, 1]
        self.camera_cx = K_matrix[0, 2]
        self.camera_cy = K_matrix[1, 2]
        self.destroy_subscription(self.sub_camera_info)
        self.get_logger().info("SUCCESSFULLY GOT CAMERA INFO")

    def color_callback(self, msg):
        # try:
        self.last_color_frame = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        # except Exception as e:
        #     self.get_logger().error(f"COLOR ERROR: {e}")
    
    def depth_callback(self, msg):
        # try:
        depth_frame = self.cv_bridge.imgmsg_to_cv2(msg, msg.encoding)   # 16UC1
        self.last_depth_frame = depth_frame

        if self.last_color_frame is not None and self.last_depth_frame is not None:
            self.process_frame()

        # except Exception as e:
            # self.get_logger().error(f"DEPTH ERROR: {e}")

    def process_frame(self):
        color_im = np.copy(self.last_color_frame)
        depth_im = np.copy(self.last_depth_frame)

        results = self.hands.process(cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                if self.show_camera_processing:
                    self.mp_draw.draw_landmarks(
                        color_im, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                
                # Извлечение ключевых точек для предсказания жеста
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                landmarks = np.array(landmarks).reshape(1, -1)

                # Если в кадре видна не вся рука, но алгоритм уже предсказал руку за пределами кадра,
                # пропускаем такой кадр, иначе ломается алгоритм
                if landmarks[landmarks > 1.0].size > 0:
                    continue
                
                # Нормализация и предсказание
                # landmarks_norm = self.scaler.transform(landmarks)
                landmarks_norm = landmarks
                gesture_id = np.argmax(self.model.predict(landmarks_norm)) + 1
                
                # Визуализация
                if self.show_camera_processing:
                    cv2.putText(
                        color_im, f"Gesture: {gesture_id} ({self.sign_names[gesture_id]})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                
                # Управление дроном
                throttle, roll, pitch, yaw = self.calculate_angles(hand_landmarks, depth_im)
                t, p, r, y = self.publish_command(gesture_id, throttle, roll, pitch, yaw)
                t, p, r, y = round(t, 2), round(p, 2), round(r, 2), round(y, 2)

                if self.show_camera_processing:
                    cv2.putText(
                        color_im, f"Throttle = {t}, pitch = {p}, roll = {r}, yaw = {y}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
        else:
            if self.show_camera_processing:
                cv2.putText(
                    color_im, f"NO CONTROL", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
            self.publish_command(   # Дрон висит в воздухе если нет команд
                1, 
                10 * (self.max_palm_height + self.min_palm_height) / 2, 
                0, 0, 0
            )

        if self.show_camera_processing:
            cv2.imshow("Hand Tracking", color_im)

        cv2.imshow("Control Panel", self.control_panel)

        if self.throttle_plot is not None:
            cv2.imshow("Throttle watcher", self.throttle_plot)
        if self.roll_plot is not None:
            cv2.imshow("Roll watcher", self.roll_plot)
        if self.pitch_plot is not None:
            cv2.imshow("Pitch watcher", self.pitch_plot)
        if self.yaw_plot is not None:
            cv2.imshow("Yaw watcher", self.yaw_plot)

        cv2.waitKey(1)

    def find_3d_angle(self, A: np.array, B: np.array, dA: np.array, dB: np.array, degrees=True) -> float:
        """
        Находит угол наклона прямой AB к камере.
            A, B - 2D-координаты точек на изображении,
            dA, dB - значения глубины в указанных координатах.
        """
        x_A = (A[0] - self.camera_cx) / self.camera_fx
        y_A = (A[1] - self.camera_cy) / self.camera_fy
        x_B = (B[0] - self.camera_cx) / self.camera_fx
        y_B = (B[1] - self.camera_cy) / self.camera_fy

        A_vector = np.array([x_A * dA, y_A * dA, dA])
        B_vector = np.array([x_B * dB, y_B * dB, dB])
        AB_vector = B_vector - A_vector

        angle = np.arctan(AB_vector[2] / np.sqrt(AB_vector[0]**2 + AB_vector[1]**2))
        
        if degrees:
            angle = np.degrees(angle)
        
        return float(angle)
    
    def normalize_value(self, value: float, min_value: float, max_value: float) -> float:
        return 2 * (value - min_value) / (max_value - min_value) - 1
    
    def calculate_angles(self, hand_landmarks, depth_im):
        """
        Возвращает:
            газ как расстояние до центра ладони в миллиметрах;
            крен, тангаж как разницу между ключевыми точками в миллиметрах;
            рыскание - наклон руки относительно камеры в градусах
        """
        depth_size = np.array(depth_im.shape)[::-1] # Чтобы было width, height

        # Ключевые точки для расчетов
        wrist = hand_landmarks.landmark[0]
        index_finger = hand_landmarks.landmark[8]
        index_finger_base = hand_landmarks.landmark[5]
        middle_finger = hand_landmarks.landmark[12]
        middle_finger_base = hand_landmarks.landmark[9]
        pinkie_finger_base = hand_landmarks.landmark[17]

        # Перевод в нампай массивы
        wrist = (np.array([wrist.x, wrist.y]) * depth_size).astype(int)
        index_finger = (np.array([index_finger.x, index_finger.y]) * depth_size).astype(int)
        index_finger_base = (np.array([index_finger_base.x, index_finger_base.y]) * depth_size).astype(int)
        middle_finger = (np.array([middle_finger.x, middle_finger.y]) * depth_size).astype(int)
        middle_finger_base = (np.array([middle_finger_base.x, middle_finger_base.y]) * depth_size).astype(int)
        pinkie_finger_base = (np.array([pinkie_finger_base.x, pinkie_finger_base.y]) * depth_size).astype(int)

        # Векторы от запястья и средняя точка
        middle_point = (wrist + index_finger_base + pinkie_finger_base) / 3

        # Расчет крена
        index_finger_base_depth = depth_im[int(index_finger_base[1]), int(index_finger_base[0])].astype(float)
        pinkie_finger_base_depth = depth_im[int(pinkie_finger_base[1]), int(pinkie_finger_base[0])].astype(float)
        roll = self.find_3d_angle(
            index_finger_base, pinkie_finger_base, 
            index_finger_base_depth, pinkie_finger_base_depth
        )

        # Расчет тангажа
        avg_finger_base_point = (index_finger_base + pinkie_finger_base) / 2
        avg_finger_base_point_depth = depth_im[int(avg_finger_base_point[1]), int(avg_finger_base_point[0])].astype(float)
        wrist_depth = depth_im[int(wrist[1]), int(wrist[0])].astype(float)
        pitch = self.find_3d_angle(
            avg_finger_base_point, wrist,
            avg_finger_base_point_depth, wrist_depth
        )

        # Расчет рыскания
        avg_finger_point = (index_finger + middle_finger) / 2
        avg_finger_base_point = (index_finger_base + middle_finger_base) / 2
        dx, dy = avg_finger_point - avg_finger_base_point
        rads = math.atan2(dy, dx)
        degs = math.degrees(rads)
        if (degs < 0):
            degs += 90
        yaw = round(degs, 1)

        # Расчет газа (throttle), Z-координата середины ладони
        throttle = depth_im[int(middle_point[1]), int(middle_point[0])].astype(float)
        
        return throttle, roll, pitch, yaw
    
    def calculate_control(self, throttle, roll, pitch, yaw):

        # Если углы малы, игнорируем их
        roll = roll if abs(roll) > self.min_angle_degrees else 0
        pitch = pitch if abs(pitch) > self.min_angle_degrees else 0
        yaw = yaw if abs(yaw) > self.min_angle_degrees else 0

        # Насыщение (ограничение) углов
        roll = np.clip(roll, -self.max_angle_degrees, self.max_angle_degrees)
        pitch = np.clip(pitch, -self.max_angle_degrees, self.max_angle_degrees)
        yaw = np.clip(yaw, -self.max_angle_degrees, self.max_angle_degrees)
        
        # Нормализация газа от -1 до 1, где <0 - снижение, >0 - подъем
        throttle = throttle / 10    # мм -> см
        throttle = self.normalize_value(throttle, self.min_palm_height, self.max_palm_height)
        throttle = throttle if abs(throttle) > 0.1 else 0  # Если газ мал, игнорируем его
        throttle = np.clip(throttle, -self.speed, self.speed)   # Насыщение (ограничение) газа

        # Нормализация крена, тангажа, рыскания от -1 до 1
        roll = self.normalize_value(roll, -self.max_angle_degrees, self.max_angle_degrees)
        pitch = self.normalize_value(pitch, -self.max_angle_degrees, self.max_angle_degrees)
        yaw = self.normalize_value(yaw, -self.max_angle_degrees, self.max_angle_degrees)

        # Сглаживание управления
        filtered_roll = self.roll_filter.update(roll)
        filtered_pitch = self.pitch_filter.update(pitch)
        filtered_yaw = self.yaw_filter.update(yaw)
        filtered_throttle = self.throttle_filter.update(throttle)

        # Мониторинг управления
        if self.control_monitoring_on:
            self.throttle_plot = self.throttle_watcher.update(throttle, filtered_throttle)
            self.roll_plot = self.roll_watcher.update(roll, filtered_roll)
            self.pitch_plot = self.pitch_watcher.update(pitch, filtered_pitch)
            self.yaw_plot = self.yaw_watcher.update(yaw, filtered_yaw)

        return filtered_throttle, filtered_roll, filtered_pitch, filtered_yaw
    
    def publish_command(self, gesture_id, throttle, roll, pitch, yaw):
        cmd = Twist()

        # Управление скоростью
        # if gesture_id in [1, 2, 3, 4, 5]:
        #     self.speed = gesture_id

        throttle, roll, pitch, yaw = self.calculate_control(
            throttle, roll, pitch, yaw
        )

        # Визуализация управления
        self.update_control_panel(throttle, roll, pitch, yaw)

        # Передача управления на коптер
        cmd.linear.z = float(throttle)
        cmd.linear.x = float(pitch)
        cmd.linear.y = float(roll) if not self.simulation else -float(roll)
        cmd.angular.z = float(yaw)

        self.cmd_pub.publish(cmd)

        return cmd.linear.z, cmd.linear.x, cmd.linear.y, cmd.angular.z
    
    def update_control_panel(self, throttle, roll, pitch, yaw) -> None:
        def draw_crosshair(img, center, size, x_label, y_label, x_val, y_val):
            cx, cy = center
            half_size = size // 2
            
            # Стиль осей
            axis_color = (0, 0, 255)  # Красный
            thickness = 3
            
            # Горизонтальная ось (X)
            cv2.line(img, (cx - half_size, cy), (cx + half_size, cy), axis_color, thickness)
            cv2.arrowedLine(img, (cx, cy), (cx + half_size, cy), axis_color, thickness, tipLength=0.15)
            cv2.arrowedLine(img, (cx, cy), (cx - half_size, cy), axis_color, thickness, tipLength=0.15)
            
            # Вертикальная ось (Y)
            cv2.line(img, (cx, cy - half_size), (cx, cy + half_size), axis_color, thickness)
            cv2.arrowedLine(img, (cx, cy), (cx, cy - half_size), axis_color, thickness, tipLength=0.15)
            cv2.arrowedLine(img, (cx, cy), (cx, cy + half_size), axis_color, thickness, tipLength=0.15)
            
            # Настройки подписей
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            text_color = (0, 0, 255)  # Красный
            
            # Подпись горизонтальной оси (X) - снизу по центру
            (x_label_width, x_label_height), _ = cv2.getTextSize(x_label, font, font_scale, font_thickness)
            cv2.putText(img, x_label,
                        (cx + half_size - x_label_width, cy + x_label_height * 3),
                        font, font_scale, text_color, font_thickness)
            
            # Подпись вертикальной оси (Y) - справа от стрелки
            (y_label_width, y_label_height), _ = cv2.getTextSize(y_label, font, font_scale, font_thickness)
            cv2.putText(img, y_label,
                        (cx + y_label_width // 2, cy - half_size + y_label_height),
                        font, font_scale, text_color, font_thickness)
            
            # Зеленый маркер с обводкой
            marker_x = cx + int(x_val * half_size)
            marker_y = cy - int(y_val * half_size)
            cv2.circle(img, (marker_x, marker_y), 15, (255, 255, 255), 3)  # Обводка
            cv2.circle(img, (marker_x, marker_y), 15, (0, 255, 0), -1)     # Заливка
            
            # Цифровые значения (под перекрестием)
            value_text = f"({x_val:.2f}, {y_val:.2f})"
            cv2.putText(img, value_text, 
                        (cx - 60, cy + half_size + 40), 
                        font, 0.6, (300, 300, 300), 1)
            
        img = np.zeros(
                (self.control_panel_height, self.control_panel_width, 3), 
                dtype=np.uint8
            )
        
        # Параметры для перекрестий
        crosshair_size = int(0.8 * self.control_panel_size) # Размер перекрестий
        padding_x = int(0.05 * self.control_panel_size)      # Боковые отступы
        padding_y = int(0.1 * self.control_panel_size)      # Вертикальные отступы
        
        # Первое перекрестие (Yaw/Throttle) - слева
        cross1_center = (padding_x + crosshair_size // 2, padding_y + crosshair_size // 2)
        draw_crosshair(img, cross1_center, crosshair_size, 'Yaw', 'Throttle', yaw, throttle)

        # Разделительная линия
        cv2.line(img, (self.control_panel_width // 2, padding_y // 2), 
                (self.control_panel_width // 2, self.control_panel_height - padding_y // 2), 
                (100, 100, 100), 2)
        
        # Второе перекрестие (Roll/Pitch) - справа
        cross2_center = (self.control_panel_width - padding_x - crosshair_size // 2, padding_y + crosshair_size // 2)
        draw_crosshair(img, cross2_center, crosshair_size, 'Roll', 'Pitch', roll, pitch)
        
        self.control_panel = img


def main():
    rclpy.init()
    node = HandGestureDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()