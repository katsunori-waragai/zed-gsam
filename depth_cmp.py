"""
depth画像を見るサンプルスクリプト

"""

import pyzed.sl as sl
import argparse

import cv2
import numpy as np

import skimage
import matplotlib.pyplot as plt

import zedhelper.handmark
import zedhelper.util
from zedhelper import predefined
import common

import inspect

MAX_ABS_DEPTH, MIN_ABS_DEPTH = 0.0, 5.0  # [m]


def resize_image(image: np.ndarray, rate: float) -> np.ndarray:
    H, W = image.shape[:2]
    return cv2.resize(image, (int(W * rate), int(H * rate)))


def as_matrix(chw_array):
    H_, W_ = chw_array.shape[-2:]
    return np.reshape(chw_array, (H_, W_))

def depth_with_hue_segment(depth_for_display_cvimg: np.ndarray, masks_cpu: np.ndarray) -> np.ndarray:
    import hsv_view

    masks_cpu = as_matrix(masks_cpu)
    depth_for_display_gray = depth_for_display_cvimg[:, :, 0]
    hsv_img = hsv_view.gen_hsv_image(depth_for_display_gray, masks_cpu)
    bgr = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    skimage.io.imsave("bgr.png", bgr)
    return bgr

def any_isnan(array: np.ndarray) -> bool:
    return np.any(np.isnan(array.flatten()))

def all_isfinite(array: np.ndarray) -> bool:
    return np.all(np.isfinite(array.flatten()))

def main(opt):
    prompt = "bottle . person . box"
    prompt = "bottle"
    watching_obj = "bottle"
    assert prompt.find(watching_obj) > -1

    zed = sl.Camera()

    init_params = predefined.InitParameters()

    common.parse_args(opt, init_params)

    init_params.depth_mode = sl.DEPTH_MODE.ULTRA

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(err)
        exit(1)

    zedhelper.util.show_params(init_params)

    depth_map = sl.Mat()
    point_cloud = sl.Mat()

    runtime_parameters = predefined.RuntimeParameters()
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    # runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
    runtime_parameters.confidence_threshold = opt.confidence_threshold
    print(f"### {runtime_parameters.confidence_threshold=}")
    zedhelper.util.show_params(runtime_parameters)

    condition_str = f"mode: {init_params.depth_mode} conf: {runtime_parameters.confidence_threshold}"

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)  # Retrieve depth
            depth_map_data = depth_map.get_data()

            # 空間座標を得ることが必要。
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            points = point_cloud.get_data()
            print(f"{points.shape=}")

            # 点群の色情報が有効な領域をvalid_points_maskとして取得する。
            points_color = points[:, :, 3]
            valid_points_mask = np.isfinite(points_color)
            print(f"{valid_points_mask.shape=} {valid_points_mask.dtype=}")
            # points[y, x]で、元画像上の点と対応がつくのかどうか？

            depth_map_data_modified = depth_map_data.copy()
            print(f"{depth_map_data_modified.shape=} {depth_map_data_modified.dtype=}")
            depth_map_data_modified[np.logical_not(valid_points_mask)] = np.nan
            plt.figure(condition_str, figsize=(16, 8))
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(depth_map_data_modified, vmax=2.0, vmin=0.0)  # far is positive
            plt.title("valid color case")
            plt.colorbar()
            plt.grid(True)
            plt.subplot(1, 2, 2)
            plt.imshow(depth_map_data, vmax=2.0, vmin=0.0)  # far is positive
            plt.title("valid depth_map case")
            plt.colorbar()
            plt.grid(True)
            plt.draw()
            plt.pause(0.001)

    depth_map.free(memory_type=sl.MEM.CPU)
    point_cloud.free(memory_type=sl.MEM.CPU)
    zed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="depth map viewer")
    parser.add_argument(
        "--input_svo_file",
        type=str,
        help="Path to an .svo file, if you want to replay it",
        default="",
    )
    parser.add_argument(
        "--ip_address",
        type=str,
        help="IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup",
        default="",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        help="Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA",
        default="",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="depth confidence_threshold(0 ~ 100)",
        default=100,
    )
    opt = parser.parse_args()
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main(opt)
