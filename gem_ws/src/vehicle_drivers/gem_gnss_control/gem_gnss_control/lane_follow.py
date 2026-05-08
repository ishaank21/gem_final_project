#!/usr/bin/env python3

import os
import math
import json
import time
import numpy as np
import torch
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pacmod2_msgs.msg import PositionWithSpeed, SystemCmdFloat, VehicleSpeedRpt

from gem_gnss_control.lane_line_fit import (
    lane_fit, final_viz, perspective_transform, closest_point_on_polynomial
)
from gem_gnss_control.lane_model_utils import load_model, inference
from gem_gnss_control.lane_worldgt import WorldGT


class LaneFollow(Node):
    def __init__(self):
        super().__init__('lane_follow_node')

        self.declare_parameter('data_dir', '/home/gem/ABO_WS/lane_follow_controller')
        self.declare_parameter('target_speed', 2.0)
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('wheelbase', 2.57)

        data_dir      = self.get_parameter('data_dir').value
        self._target_speed = self.get_parameter('target_speed').value
        image_topic   = self.get_parameter('image_topic').value
        self.L        = self.get_parameter('wheelbase').value

        # Load perception model
        self._dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self._model = load_model(data_dir).to(self._dev).eval()
            self.get_logger().info('SimpleEnet model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Could not load SimpleEnet model: {e}')
            raise

        # Load BEV calibration
        try:
            with open(os.path.join(data_dir, 'data', 'bev_config.json')) as f:
                self._bev_cfg = json.load(f)
        except Exception as e:
            self.get_logger().error(f'Could not load bev_config.json: {e}')
            raise

        # Ground truth helper — debug only, fails silently if unavailable
        try:
            resources_dir = os.path.join(data_dir, 'resources')
            self._world = WorldGT('HighBay', resources_dir=resources_dir)
        except Exception as e:
            self.get_logger().warn(f'WorldGT unavailable (GT metrics disabled): {e}')
            self._world = None

        self._cv_bridge = CvBridge()
        self._current_speed = 0.0
        self._pacmod_enable = False

        # Subscriptions
        self.create_subscription(Image, image_topic, self._on_image, 10)
        self.create_subscription(Bool, '/pacmod/enabled', self._on_enable, 10)
        self.create_subscription(VehicleSpeedRpt, '/pacmod/vehicle_speed_rpt', self._on_speed, 10)

        # Publishers
        self._steer_pub = self.create_publisher(PositionWithSpeed, '/pacmod/steering_cmd', 1)
        self._accel_pub = self.create_publisher(SystemCmdFloat, '/pacmod/accel_cmd', 1)
        self._brake_pub = self.create_publisher(SystemCmdFloat, '/pacmod/brake_cmd', 1)

        self.get_logger().info(
            f'LaneFollow ready — image: {image_topic}, speed: {self._target_speed} m/s, '
            f'wheelbase: {self.L} m'
        )

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #

    def _on_enable(self, msg):
        self._pacmod_enable = msg.data

    def _on_speed(self, msg):
        self._current_speed = msg.vehicle_speed

    def _on_image(self, msg):
        image = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

        mask = inference(self._model, image, self._dev)
        m = mask.astype(np.uint8) * 255

        combine_fit_img, binary_BEV, ret = self._fit_poly_lanes(image, m)
        binary_BEV = np.pad(binary_BEV, ((0, 100), (0, 0)))
        binary_BEV = cv2.cvtColor(binary_BEV, cv2.COLOR_GRAY2BGR)

        if ret:
            left_fit  = ret['left_fit']
            right_fit = ret['right_fit']

            # Pick lookahead point at y=350; fall back to y=500 if lanes are too close
            lookahead_y_px = 350
            left_x  = np.polyval(left_fit,  lookahead_y_px)
            right_x = np.polyval(right_fit, lookahead_y_px)
            if (right_x - left_x) < 80:
                lookahead_y_px = 500
                left_x  = np.polyval(left_fit,  lookahead_y_px)
                right_x = np.polyval(right_fit, lookahead_y_px)

            target_x_px = (left_x + right_x) / 2.0

            Sy, Sx = self._bev_cfg['unit_conversion_factor']
            x_forward_m  = (600 - lookahead_y_px) * Sy
            y_lateral_m  = -(target_x_px - 400.0) * Sx

            # Pure pursuit → front wheel angle
            ld = math.hypot(x_forward_m, y_lateral_m)
            if ld > 0.001:
                alpha       = math.atan2(y_lateral_m, x_forward_m)
                front_angle = math.atan2(2 * self.L * math.sin(alpha), ld)
            else:
                front_angle = 0.0
            front_angle = float(np.clip(front_angle, -0.61, 0.61))

            if self._pacmod_enable:
                steer_cmd = PositionWithSpeed()
                steer_cmd.angular_position      = self._front2steer(front_angle)
                steer_cmd.angular_velocity_limit = 4.0
                self._steer_pub.publish(steer_cmd)

                speed_err = self._target_speed - self._current_speed
                throttle  = max(0.0, min(0.3, 0.15 * speed_err))
                self._accel_pub.publish(SystemCmdFloat(command=throttle))
                self._brake_pub.publish(SystemCmdFloat(command=0.0))

            self.get_logger().info(
                f'steer={math.degrees(front_angle):.1f}°  '
                f'speed={self._current_speed:.2f}/{self._target_speed:.1f} m/s',
                throttle_duration_sec=0.5
            )

            # Visualization overlay
            poly_px = np.add(left_fit, right_fit) / 2
            ploty   = ret['ploty']
            cv2.polylines(binary_BEV,
                [np.stack((np.polyval(left_fit,  ploty), ploty), 1).astype(np.int32)],
                False, (255, 0, 0), 4)
            cv2.polylines(binary_BEV,
                [np.stack((np.polyval(poly_px,   ploty), ploty), 1).astype(np.int32)],
                False, (0, 255, 255), 4)
            cv2.polylines(binary_BEV,
                [np.stack((np.polyval(right_fit, ploty), ploty), 1).astype(np.int32)],
                False, (0, 0, 255), 4)
            cv2.circle(binary_BEV, (int(target_x_px), int(lookahead_y_px)), 12, (255, 0, 255), -1)

        else:
            if self._pacmod_enable:
                self.get_logger().warn('No lanes detected — stopping vehicle')
                self._publish_stop()

        cv2.imshow('lane_bev',     binary_BEV)
        cv2.imshow('lane_overlay', combine_fit_img if combine_fit_img is not None else image)
        cv2.waitKey(1)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _front2steer(self, front_angle_rad: float) -> float:
        """GEM E4 front wheel angle (rad) → steering wheel position (rad)."""
        deg     = math.degrees(front_angle_rad)
        deg     = max(min(deg, 35.0), -35.0)
        abs_deg = abs(deg)
        steer   = -0.1084 * abs_deg**2 + 21.775 * abs_deg
        if deg < 0:
            steer = -steer
        return math.radians(steer)

    def _publish_stop(self):
        self._brake_pub.publish(SystemCmdFloat(command=0.5))
        self._accel_pub.publish(SystemCmdFloat(command=0.0))

    def _fit_poly_lanes(self, raw_img, binary_img):
        binary_warped, M, Minv = perspective_transform(
            binary_img, np.float32(self._bev_cfg['src'])
        )
        ret = lane_fit(binary_warped)
        if ret is None:
            return None, binary_warped, None
        combine_fit_img = final_viz(raw_img, ret['left_fit'], ret['right_fit'], Minv)
        return combine_fit_img, binary_warped, ret


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollow()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down — applying brakes')
        node._publish_stop()
        time.sleep(0.5)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
