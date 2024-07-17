from setuptools import find_packages, setup

package_name = 'mediapipe_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/video', ['video/test_video.mp4']),
        ('share/' + package_name + '/model', ['model/pose_landmarker_full.task']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_estimation_node = mediapipe_ros.pose_estimation_node:main',
            'visualization_node = mediapipe_ros.visualization:main',
            'video_publisher_node = mediapipe_ros.video_publisher_node:main',
        ],
    },
)
