import os

import torch
import json
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.parameter import Parameter

from worldgt import WorldGT
from line_fit import lane_fit, final_viz, perspective_transform, closest_point_on_polynomial
from model_utils import load_model, inference
import rich
import cv2
from scipy.spatial.transform import Rotation as R


class LaneVisualizer(Node):
    def __init__(self):
        super().__init__("lane_visualizer")

        sim_time_param = Parameter('use_sim_time', Parameter.Type.BOOL, True)
        self.set_parameters([sim_time_param])

        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._model = load_model()
            if self._model is not None:
                self._model = self._model.to(self._dev)
                self._model = self._model.eval()
                rich.print("[green]loaded SimpleEnet :o")
            else:
                self.get_logger().error(f"could not load SimpleEnet model x_X: {e}")
                exit(1)
        except Exception as e:
            self.get_logger().error(f"could not load SimpleEnet model x_X: {e}")
            exit(1)
        
        try: 
            with open(os.path.join("data", "bev_config.json")) as f:
                self._bev_cfg = json.load(f)
        except FileNotFoundError:
            self.get_logger().error(f"could not load bev config x_X: {e}")
            exit(1)

        self._world = WorldGT("HighBay")
        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)
        
        self._image_msg = None
        self._cv_bridge = CvBridge()
        
        self.create_subscription(
            Image,
            "/camera/image_raw",
            self._on_image,
            10
        )

    def _on_image(self, msg) -> None:
            self._image_msg = msg
            if self._model is None:
                return
            
            # 1. Initialize variables with default values to prevent UnboundLocalError
            XTE, HE = "N/A", "N/A"
            lane, gt_XTE, gt_HE = "unknown", "N/A", "N/A"
            
            # 2. Convert and crop the stereo image (Taking only the LEFT half for ZED bags)
            full_image = self._cv_bridge.imgmsg_to_cv2(self._image_msg, "bgr8")
            height, width, _ = full_image.shape
            image = full_image[:, :width//2, :] 
            
            # 3. Perception Pipeline: Run inference to obtain binary mask
            mask = inference(self._model, image, self._dev)
            m = mask.astype(np.uint8) * 255

            # 4. Noise Filtering: Morphological Operations
            # Use a kernel to scrub small blobs (Opening) and solidify lane lines (Closing)
            kernel = np.ones((5, 5), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

            # 5. Spatial Masking: Horizon and Side-Masks
            h_mask, w_mask = m.shape[:2]
            # Horizon: Zero out top 62% to hide buildings and trucks
            m[:int(h_mask * 0.62), :] = 0 
            # Side-Masking: Zero out left and right 15% to stop IPM streaking
            m[:, :int(w_mask * 0.15)] = 0
            m[:, int(w_mask * 0.85):] = 0
            
            # 6. Lane Fitting and BEV Projection
            combine_fit_img, binary_BEV, ret = self.fit_poly_lanes(image, m)

            # Prepare BEV for drawing
            binary_BEV = np.pad(binary_BEV, ((0, 100), (0, 0)))
            binary_BEV = cv2.cvtColor(binary_BEV, cv2.COLOR_GRAY2BGR)
            
            # 7. Process fit results if lanes were successfully detected
            if ret:                
                poly_px = (np.add(ret["left_fit"], ret["right_fit"]) / 2)
                est_xte_val, est_he_val, camera_px, closest_px = self.compute_error(poly_px)
                
                # Draw lane lines on the BEV
                ploty = ret['ploty']
                left_fitx = np.polyval(ret["left_fit"], ploty)
                center_fitx = np.polyval(poly_px, ploty)
                right_fitx = np.polyval(ret["right_fit"], ploty)
                
                pts_left = np.stack((left_fitx, ploty), axis=1).astype(np.int32)
                pts_center = np.stack((center_fitx, ploty), axis=1).astype(np.int32)
                pts_right = np.stack((right_fitx, ploty), axis=1).astype(np.int32)

                cv2.polylines(binary_BEV, [pts_center], isClosed=False, color=(0, 255, 255), thickness=4)                
                cv2.polylines(binary_BEV, [pts_left], isClosed=False, color=(255, 0, 0), thickness=4)
                cv2.polylines(binary_BEV, [pts_right], isClosed=False, color=(0, 0, 255), thickness=4)

                # Draw XTE/HE visualization markers
                cv2.circle(binary_BEV, (int(closest_px[0]), int(closest_px[1])), 8, (0, 255, 0), -1)
                cv2.line(binary_BEV, (int(camera_px[0]), int(camera_px[1])), (int(closest_px[0]), int(closest_px[1])), (0, 255, 0), 4)

                # Format outputs for terminal display
                XTE = f"{est_xte_val:.2f}"
                HE = f"{np.degrees(est_he_val):.2f}"

            # 8. Attempt Ground Truth (GT) lookup from TF tree
            try:
                trans = self._tf_buf.lookup_transform("highbay", "stereo_camera_link", msg.header.stamp)
                pos = trans.transform.translation
                q = trans.transform.rotation
                rotation = R.from_quat([q.x, q.y, q.z, q.w])
                yaw = rotation.as_euler('xyz', degrees=False)[2]
                lane, _, gt_xte_val, gt_he_val = self._world.get_metrics(pos.x, pos.y, yaw)
                gt_XTE = f"{gt_xte_val:.2f}"
                gt_HE = f"{np.degrees(gt_he_val):.2f}"
            except:
                pass # Keep default "N/A" values set at the top
                
            # 9. Visualization and Console Log
            print(f"EST XTE: {XTE} m - HE: {HE}° -- GT XTE: {gt_XTE} m HE: {gt_HE}° - lane: {lane}")

            if combine_fit_img is None:
                combine_fit_img = image

            cv2.imshow("render_view", combine_fit_img)
            cv2.imshow("binary_BEV", binary_BEV)
            cv2.waitKey(1)
    
    # def _on_image(self, msg) -> None:
    #     self._image_msg = msg
    #     if self._model is None:
    #         return
        
    #     image = self._cv_bridge.imgmsg_to_cv2(self._image_msg, "bgr8")
    #     mask = inference(self._model, image, self._dev)
    #     m = mask.astype(np.uint8) * 255

    #     # Adding in to clean up binary mask

    #     # kernel = np.ones((5, 5), np.uint8)
    #     # m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    #     # m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    #     # h = m.shape[0]
    #     # m[:int(h*0.5), :] = 0   # zero out top 50%
    #     ##
    #     combine_fit_img, binary_BEV, ret = self.fit_poly_lanes(image, m)

    #     binary_BEV = np.pad(binary_BEV, ((0, 100), (0, 0)))
    #     binary_BEV = cv2.cvtColor(binary_BEV, cv2.COLOR_GRAY2BGR)
        
    #     if ret:                
    #         poly_px = (np.add(ret["left_fit"], ret["right_fit"]) / 2)
    #         XTE, HE, camera_px, closest_px = self.compute_error(poly_px)
            
    #         # draw lane lines
    #         ploty = ret['ploty']
    #         left_fitx = np.polyval(ret["left_fit"], ploty)
    #         center_fitx = np.polyval(poly_px, ploty)
    #         right_fitx = np.polyval(ret["right_fit"], ploty)
            
    #         pts_left = np.stack((left_fitx, ploty), axis=1).astype(np.int32)
    #         pts_center = np.stack((center_fitx, ploty), axis=1).astype(np.int32)
    #         pts_right = np.stack((right_fitx, ploty), axis=1).astype(np.int32)

    #         cv2.polylines(binary_BEV, [pts_center], isClosed=False, color=(0, 255, 255), thickness=4)                
    #         cv2.polylines(binary_BEV, [pts_left], isClosed=False, color=(255, 0, 0), thickness=4)
    #         cv2.polylines(binary_BEV, [pts_right], isClosed=False, color=(0, 0, 255), thickness=4)

    #         # draw closest point and bridge line
    #         cv2.circle(binary_BEV, (int(closest_px[0]), int(closest_px[1])), 8, (0, 255, 0), -1)
    #         cv2.line(
    #             binary_BEV,
    #             (int(camera_px[0]), int(camera_px[1])),
    #             (int(closest_px[0]), int(closest_px[1])),
    #             (0, 255, 0),
    #             4
    #         )

    #         # draw camera chevron
    #         cv2.line(
    #             binary_BEV,
    #             (int(camera_px[0]), int(camera_px[1])),
    #             (int(camera_px[0] - 20), int(camera_px[1] + 20)),
    #             (255, 0, 255),
    #             4
    #         )
    #         cv2.line(
    #             binary_BEV,
    #             (int(camera_px[0]), int(camera_px[1])),
    #             (int(camera_px[0] + 20), int(camera_px[1] + 20)),
    #             (255, 0, 255),
    #             4
    #         )

    #         XTE = f"{XTE:.2f}"
    #         HE = f"{np.degrees(HE):.2f}"
    #     else:
    #         XTE = "N/A"
    #         HE = "N/A"

    #     try:
    #         trans = self._tf_buf.lookup_transform("highbay", "stereo_camera_link", msg.header.stamp)
    #         pos = trans.transform.translation
    #         q = trans.transform.rotation
    #         rotation = R.from_quat([q.x, q.y, q.z, q.w])
    #         euler_angles = rotation.as_euler('xyz', degrees=False)
    #         yaw = euler_angles[2]
    #         lane, _, gt_XTE, gt_HE = self._world.get_metrics(pos.x, pos.y, yaw)
    #         gt_XTE = f"{gt_XTE:.2f}"
    #         gt_HE = f"{np.degrees(gt_HE):.2f}"
    #     except:
    #         lane = "unknown"
    #         gt_XTE = "N/A"
    #         gt_HE = "N/A"
            
    #     print(f"EST XTE: {XTE} m - HE: {HE}° -- GT XTE: {gt_XTE} m HE: {gt_HE}° - lane: {lane}")

    #     if combine_fit_img is None:
    #         combine_fit_img = image
            

    #     cv2.imshow("render_view", combine_fit_img)
    #     cv2.imshow("binary_BEV", binary_BEV)
    #     cv2.waitKey(1)

    
    def compute_error(self, poly_px):
        """
        Calculates Cross-Track Error (XTE) and Heading Error.

        poly_px:    polynomial coefficients defined in pixels
                    ex for 2nd order: (A, B, C) where x = Ay^2 + By + C
        """
        bev_height_m, bev_width_m = self._bev_cfg["bev_world_dim"]
        Sy, Sx = self._bev_cfg["unit_conversion_factor"]
        scale = np.array([Sx, Sy])

        camera_m = np.array([(bev_width_m / 2), bev_height_m])
        camera_px = camera_m / scale
        closest_px = closest_point_on_polynomial(camera_px, poly_px)
        closest_m = closest_px * scale

        ##### YOUR CODE STARTS HERE #####

        # calculate cross track error
        # hint: |XTE| = distance between camera and closest point
        #       on ploly_px however XTE is not a strictly positive value
        # XTE = np.sqrt(((camera_m[0] - closest_m[0]) ** 2) + ((camera_m[1] - closest_m[1]) ** 2)) # Euclidean Distance between car and lane
        # if (camera_m[0] > closest_m[0]):
        #     XTE *=-1

        XTE = -camera_m[0] + closest_m[0]

        # hint: find derivative of the polynomial at the closest point
        #       then use arctan on the scaled slope
        # HE = 0

        # der_poly = []
        # for i in range (len(poly_px) - 1): # Computes derivative and saves in a list
        #     der_poly.append(poly_px[i] * (len(poly_px) - i - 1))

        # # Plug in closest point
        # sum_val = 0

        # for i in range(len(der_poly)):
        #     sum_val += der_poly[len(der_poly) - i - 1] * (closest_px[1] ** i)

        # sum_val *= Sx/Sy
        # HE = np.arctan(sum_val)

        A, B, C = poly_px

        y_closest_px = closest_px[1]
        slope_px = 2*A*y_closest_px + B
        HE = np.arctan2(slope_px*Sx, Sy)


        ##### YOUR CODE ENDS HERE #####

        return XTE, HE, camera_px, closest_px

    def fit_poly_lanes(self, raw_img, binary_img):
        binary_warped, M, Minv = perspective_transform(binary_img, np.float32(self._bev_cfg["src"]))
        ret = lane_fit(binary_warped)
        if ret is None:
            self.get_logger().debug("ret is None; returning None for both.")
            return None, binary_warped, None
        left_fit = ret['left_fit']
        right_fit = ret['right_fit']
        
        combine_fit_img = None
        if ret is not None:
            self.get_logger().debug("Model detected lanes")
            combine_fit_img = final_viz(raw_img, left_fit, right_fit, Minv)
        else:
            self.get_logger().debug("Model unable to detect lanes")
        return combine_fit_img, binary_warped, ret


def main(args=None):
    rclpy.init(args=args)
    node = LaneVisualizer()
    rclpy.spin(node)
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()


