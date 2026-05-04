# import os
# import json
# import numpy as np


# def main():
#     # intrinsics (K) from /front_single_camera/camera_info
#     K = np.array([
#         [476.7030836014194, 0.0, 400.0],
#         [0.0, 476.7030836014194, 300.0],
#         [0.0, 0.0, 1.0]
#     ])
#     # extrinsics (R,t): base_footprint -> front_single_camera_link
#     R = np.array([
#         [0, -1, 0],
#         [ 0, 0, 1],
#         [ -1, 0, 0]
#     ])
#     t = np.array([0.160, -0.110, 1.546])

#     # bird eye view image size (px); 
#     # NOTE: DO NOT CHANGE THIS
#     bev_img_height, bev_img_width = 600, 800

#     # HYPERPARAMETERS: BEV rectangle configuration (m)
#     # NOTE: Revert this to the original values for submission
#     bev_height, bev_width = 15, 20

#     # px -> m conversion factor
#     unit_conversion_factor = (bev_height/bev_img_height, bev_width/bev_img_width)
#     bev_world_coords = np.float32([
#         [bev_height, -bev_width/2, 0],
#         [0, -bev_width/2, 0],
#         [0, bev_width/2, 0],
#         [bev_height, bev_width/2, 0],
#     ])

#     # convert the bev_world_coords into pixel coordinates
#     src = []
#     for pt in bev_world_coords:
#         ##### YOUR CODE STARTS HERE #####
#         pt_cam = R @ pt + (-R @ t)
#         pt_2d = K @ pt_cam

#         u = pt_2d[0] / pt_2d[2]
#         v = pt_2d[1] / pt_2d[2]

#         src.append([u,v])
#         ##### YOUR CODE ENDS HERE #####
#         pass
#     src = np.float32(src)

#     output = {
#         "bev_world_dim": (bev_height, bev_width),
#         "unit_conversion_factor": unit_conversion_factor,
#         "src": src.tolist(),
#     }
#     # save config to json
#     save_fn = 'data/bev_config.json'
#     if not os.path.isdir('data/'):
#         print(f"Data directory not found. Generating...")
#         os.makedirs('data/', exist_ok=False)
#     if os.path.isfile(save_fn):
#         if input("File already exists. Overwrite? (y/n):").lower() != 'y':
#             print("Exiting...")
#             import sys
#             sys.exit()
#     with open(save_fn, "w") as f:
#         json.dump(output, f, indent=2)
#     print(f"Saved BEV config to {save_fn}.")


# if __name__ == "__main__":
#     main()
import os
import json
import numpy as np

def main():
    # 1. Simulator Intrinsics (Original 800x600 values)
    K = np.array([
        [476.7030836014194, 0.0, 400.0],
        [0.0, 476.7030836014194, 300.0],
        [0.0, 0.0, 1.0]
    ])
    
    # 2. Simulator Extrinsics
    R = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    t = np.array([0.160, -0.110, 1.546])

    # 3. Simulator Dimensions (Reverting to 15x20 for the wide sim track)
    bev_height, bev_width = 15.0, 20.0
    bev_img_height, bev_img_width = 600, 800
    unit_conversion_factor = (bev_height/bev_img_height, bev_width/bev_img_width)
    
    bev_world_coords = np.float32([
        [bev_height, -bev_width/2, 0],
        [0, -bev_width/2, 0],
        [0, bev_width/2, 0],
        [bev_height, bev_width/2, 0],
    ])

    src = []
    print("--- BEV Projection Debug ---")
    for i, pt in enumerate(bev_world_coords):
        pt_cam = R @ pt + (-R @ t)
        pt_2d = K @ pt_cam
        u = pt_2d[0] / pt_2d[2]
        v = pt_2d[1] / pt_2d[2]
        
        # Valid range: u [0, 640], v [0, 360]
        print(f"World {pt} -> Pixel (u: {u:.1f}, v: {v:.1f})")
        src.append([u,v])

    output = {
        "bev_world_dim": (bev_height, bev_width),
        "unit_conversion_factor": unit_conversion_factor,
        "src": src,
    }
    
    save_fn = 'data/bev_config.json'
    if not os.path.isdir('data/'):
        os.makedirs('data/', exist_ok=False)
    with open(save_fn, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved BEV config to {save_fn}.")

if __name__ == "__main__":
    main()