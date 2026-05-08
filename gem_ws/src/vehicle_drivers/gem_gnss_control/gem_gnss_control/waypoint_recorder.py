#!/usr/bin/env python3

# ================================================================
# Waypoint Recorder Node
# Records: longitude, latitude, heading
# Subscribes: /navsatfix, /insnavgeod
# ================================================================

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from septentrio_gnss_driver.msg import INSNavGeod

import csv
import math
import os
from datetime import datetime


class WaypointRecorder(Node):

    def __init__(self):
        super().__init__('waypoint_recorder')

        # Parameters
        self.declare_parameter('record_rate', 5.0)
        self.declare_parameter('min_distance', 0.5)  # meters
        self.declare_parameter('output_file', 'track_e2.csv')

        self.record_rate = self.get_parameter('record_rate').value
        self.min_distance = self.get_parameter('min_distance').value
        self.output_file = self.get_parameter('output_file').value

        # State
        self.lat = None
        self.lon = None
        self.heading = None
        self.last_lat = None
        self.last_lon = None

        # Subscribers
        self.create_subscription(
            NavSatFix,
            '/navsatfix',
            self.gnss_callback,
            10
        )

        self.create_subscription(
            INSNavGeod,
            '/insnavgeod',
            self.ins_callback,
            10
        )

        # Timer
        self.timer = self.create_timer(
            1.0 / self.record_rate,
            self.record_waypoint
        )

        # Open CSV file
        self.file = open(self.output_file, 'w', newline='')
        self.writer = csv.writer(self.file)

        self.get_logger().info(f"Recording waypoints to {self.output_file}")
        self.get_logger().info("Press CTRL+C to stop recording")

    # ---------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------

    def gnss_callback(self, msg: NavSatFix):
        self.lat = msg.latitude
        self.lon = msg.longitude

    def ins_callback(self, msg: INSNavGeod):
        self.heading = msg.heading

    # ---------------------------------------------------------
    # Distance check (simple Haversine)
    # ---------------------------------------------------------

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi/2)**2 + \
            math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c

    # ---------------------------------------------------------
    # Record logic
    # ---------------------------------------------------------

    def record_waypoint(self):

        if self.lat is None or self.lon is None or self.heading is None:
            return

        # Skip if too close to last point
        if self.last_lat is not None:
            dist = self.haversine_distance(
                self.last_lat, self.last_lon,
                self.lat, self.lon
            )
            if dist < self.min_distance:
                return

        # Write waypoint
        self.writer.writerow([self.lon, self.lat, self.heading])
        self.file.flush()

        self.last_lat = self.lat
        self.last_lon = self.lon

        self.get_logger().info(
            f"Saved: Lon={self.lon:.8f}, "
            f"Lat={self.lat:.8f}, "
            f"Heading={self.heading:.2f}"
        )

    # ---------------------------------------------------------
    # Shutdown cleanly
    # ---------------------------------------------------------

    def destroy_node(self):
        self.get_logger().info("Closing waypoint file...")
        self.file.close()
        super().destroy_node()


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = WaypointRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()