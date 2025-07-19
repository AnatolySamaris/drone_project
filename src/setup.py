from setuptools import find_packages, setup
import os

package_name = 'cv_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", package_name, "models"), ["models/model2.h5"]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anatoly',
    maintainer_email='anatolysamaris@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "gesture_detector = cv_control.gesture_detector:main",
        ],
    },
)
