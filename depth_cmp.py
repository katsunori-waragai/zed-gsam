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

import gsam_module

import inspect

MAX_ABS_DEPTH, MIN_ABS_DEPTH = 0.0, 2.0  # [m]

def parse_args(init):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if (
            ip_str.replace(":", "").replace(".", "").isdigit()
            and len(ip_str.split(".")) == 4
            and len(ip_str.split(":")) == 2
        ):
            init.set_from_stream(ip_str.split(":")[0], int(ip_str.split(":")[1]))
            print("[Sample] Using Stream input, IP : ", ip_str)
        elif ip_str.replace(":", "").replace(".", "").isdigit() and len(ip_str.split(".")) == 4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ", ip_str)
        else:
            print("Unvalid IP format. Using live stream")
    if "HD2K" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif "HD1200" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif "HD1080" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif "HD720" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif "SVGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif "VGA" in opt.resolution:
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution) > 0:
        print("[Sample] No valid resolution entered. Using default")
    else:
        print("[Sample] Using default resolution")


def resize_image(image: np.ndarray, rate: float) -> np.ndarray:
    H, W = image.shape[:2]
    return cv2.resize(image, (int(W * rate), int(H * rate)))

def points_by_segmentation(points: np.ndarray, segmentation_image: np.ndarray):
    """
    segmentationは結果を元に、対応する点群の範囲を返す。
    points: height, width, channel の構成
    pointsは添字の順番がheight, width, channelの順番である。
    chanelには、X, Y, Z, colorが含まれている。
    segmentation_imageは、height, width のデータ
    segmentationの添字はheight, width の順番である。
    セグメンテーションの分類はuint8 の整数で分類済みである。

    なお、background に対するpointsのデータを返しても有用性が低そうなので、
    いったんは、除外することとした。

    戻り値は、各セグメンテーションに対応するpointsのsubsetのリストを返す。
    """
    # Check the dtype of the inputs
    assert points.dtype in [np.float32, np.float64], "points must be of type float32 or float64"
#    assert segmentation_image.dtype in [np.uint8, np.int], "segmentation_image must be of type uint8"

    # Check the shape of the inputs
    assert points.ndim == 3, "points must be a 3D array (height, width, channels)"
    assert segmentation_image.ndim == 2, "segmentation_image must be a 2D array (height, width)"
    assert points.shape[
           :2] == segmentation_image.shape, "points and segmentation_image must have the same height and width"

    # Get unique segmentation labels
    unique_labels = np.unique(segmentation_image)

    # Initialize a list to hold points for each segmentation label
    segmented_points = []

    # Iterate through unique labels and collect corresponding points
    for label in unique_labels:
        if label == 0:
            # 0 は background です。
            continue
        mask = segmentation_image == label
        labeled_points = points[mask]
        segmented_points.append(labeled_points)

    return segmented_points

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
    gsam_predictor = gsam_module.GroundedSAMPredictor(
        text_prompt=prompt,
        text_threshold=0.25,
        box_threshold=0.3,
        use_sam_hq=False,
    )

    zed = sl.Camera()

    init_params = predefined.InitParameters()

    parse_args(init_params)

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
            plt.figure(10, figsize=(16, 12))
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(depth_map_data_modified, vmax=2.0, vmin=0.0)  # far is positive
            plt.title("depth_map_data_modified")
            plt.colorbar()
            plt.grid(True)
            plt.subplot(1, 2, 2)
            plt.imshow(depth_map_data, vmax=2.0, vmin=0.0)  # far is positive
            plt.title("depth_map_data")
            plt.colorbar()
            plt.grid(True)
            plt.draw()
            plt.pause(0.001)
            continue

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    depth_map.free(memory_type=sl.MEM.CPU)
    point_cloud.free(memory_type=sl.MEM.CPU)
    zed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
