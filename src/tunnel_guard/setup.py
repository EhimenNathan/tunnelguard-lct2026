import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'tunnel_guard'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.json') + glob('config/*.csv')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TunnelGuard team',
    maintainer_email='ehimennathan@gmail.com',
    description='Lidar foreign-object detection inside the dynamic envelope of a driverless metro train.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_node = tunnel_guard.detector_node:main',
            'evaluate_bag = tunnel_guard.evaluate_bag:main',
        ],
    },
)
