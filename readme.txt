# ============================
# === РАЗВОРАЧИВАЕМ ПРОЕКТ ===
# ============================

# Из корня проекта (где src)
colcon build --packages-select cv_control

# Обновляем пути (выполнять в каждом рабочем терминале
source install/setup.bash

# === ====== ===
# === ЗАПУСК ===
# === ====== ===
# ВЫПОЛНЯЕМ ВСЕ КОМАНДЫ ИЗ КОРНЯ ПРОЕКТА

# Запуск симуляции с дроном
ign gazebo src/drone-world/quadcopter-teleop/track.sdf

# Запуск моста между gazebo и ros2
ros2 run ros_gz_bridge parameter_bridge /x500/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist

# Запуск камеры
ros2 launch my_realsense rs_launch.py

# Запуск ноды управления дроном
ros2 run cv_control gesture_detector
