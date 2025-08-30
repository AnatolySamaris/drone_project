# РАЗВОРАЧИВАЕМ ПРОЕКТ

## Пререквизиты:
Оборудование и софт:
- Jetson Orin NX (Ubuntu 22.04)
- Intel RealSense (В пакете my_realsense конфиг под D435i)
- DJI Tello (Для работы с физическим дроном)
- Gazebo Ignition (Для работы в симуляции)
- ROS2 Humble

## Зависимости:
Установить из apt/pip либо билдить локально. Могут быть проблемы с зависимостями, решать самостоятельно по ситуации.
- librealsense
- mediapipe >= 0.10.18
- pytorch
- djitellopy (или djitellopy2 - более старые зависимости, подходит для mediapipe == 0.10.18)

При желании использовать GPU необходимо установить mediapipe, собранный для работы с GPU, либо билдить локально:
https://www.programmerall.com/article/35072737545/

А также убедиться, что GPU вообще доступен, командой ```jtop``` (для Jetson)

В случае проблем с protobuf (связано с mediapipe), можно поставить переменную:
```
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

> Это позволит запускать mediapipe корректно, но будет работать гораздо медленнее, чем могло бы. Переменную нужно устанавливать каждый раз при создании нового терминала, из которого запускается проект

## Билдим проект
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

При желании убрать ноду ручного управления необходимо в src/launch.py закомментировать следующий код:
```
# Tello control node
Node(
    package='tello_control',
    executable='tello_control',
    namespace='/',
    name='control',
    output='screen',
    respawn=False
),
```

Однако для безопасности в процессе разработки и тестирования лучше оставить: эта нода запускает окно, принимающее команды управления дроном с клавиатуры:
- ```t``` - Takeoff
- ```l``` - Land
- ```e``` - Emergency stop
- ```w```, ```s``` - Газ вверх/вниз
- ```a```, ```d``` - Рыскание влево/вправо
- ```up```, ```down``` (стрелки) - тангаж вперед/назад
- ```left```, ```right``` (стрелки) - крен влево/вправо

Скорость реакции на управление клавиатурой задаётся в tello_control/src/main.cpp параметром ```manual_speed```

> Высота взлёта (Takeoff) в данной реализации не регулируется! Настройте её через мобильное приложение Tello (https://www.dji.com/fi/downloads/djiapp/tello) или реализуйте самостоятельно через djitellopy.

# Разработка
```
TODO
```

# LIBREALSENSE PATCHES FOR JETSON
https://github.com/jetsonhacks/jetson-orin-librealsense
