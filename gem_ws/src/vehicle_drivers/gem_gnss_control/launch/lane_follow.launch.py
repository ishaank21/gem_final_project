from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    vehicle_env = os.environ.get('VEHICLE_NAME', 'e4')
    config_path = os.path.join(
        get_package_share_directory('gem_gnss_control'),
        'config',
        f'{vehicle_env}_lf.yaml'
    )

    return LaunchDescription([
        Node(
            package='gem_gnss_control',
            executable='lane_follow',
            name='lane_follow_node',
            output='screen',
            parameters=[config_path],
            arguments=['--ros-args', '--log-level', 'info']
        )
    ])
