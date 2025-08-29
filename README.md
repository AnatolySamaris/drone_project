# РАЗВОРАЧИВАЕМ ПРОЕКТ

Клонируем репозиторий и переходим в корень проекта:
```
git clone https://github.com/AnatolySamaris/drone_project
cd drone_project
```

Далее выполняем все команды из корня проекта!

Если проект собирается впервые:
```
colcon build
source install/setup.bash
```

Если были изменения в gesture_detector.py:
```
colcon build --packages-select cv_control
source install/setup.bash
```

При каждом билде обновляем пути (выполнять в каждом рабочем терминале)
```
source install/setup.bash
```

# ЗАПУСК

## Запуск камеры Intel RealSense D435i

В отдельном терминале выполнить:
```
source install/setup.bash
ros2 launch my_realsense rs_launch.py
```

## Запуск управления

### Запуск в симуляции с дроном X500 (без автопилота)
В ```cv_control/gesture_detector.py``` ставим ```self.simulation = True```

Симуляция в газебо:
```
ign gazebo src/drone-world/quadcopter-teleop/track.sdf
```

Мост между ROS2 и Gazebo:
```
ros2 run ros_gz_bridge parameter_bridge /x500/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist
```

Нода с управлением:
```
ros2 run cv_control gesture_detector
```

### Запуск управления дроном Tello

В ```cv_control/gesture_detector.py``` должно быть ```self.simulation = False```

Запуск ноды управления, Tello драйвера (tello) и ноды ручного управления (tello_control):
```
ros2 launch src/launch.py
```

# Разработка
```
TODO
```


# LIBREALSENSE PATCHES FOR JETSON
https://github.com/jetsonhacks/jetson-orin-librealsense
