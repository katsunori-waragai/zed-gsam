"""
depth画像を見るサンプルスクリプト

"""

import pyzed.sl as sl
import argparse

import cv2
import numpy as np

from zedhelper import predefined

import inspect

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


def any_isnan(array: np.ndarray) -> bool:
    return np.any(np.isnan(array.flatten()))

def any_isneginf(array: np.ndarray) -> bool:
    return np.any(np.isneginf(array.flatten()))

def any_isposinf(array: np.ndarray) -> bool:
    return np.any(np.isposinf(array.flatten()))

def all_isfinite(array: np.ndarray) -> bool:
    return np.all(np.isfinite(array.flatten()))

def main(opt):
    zed = sl.Camera()
    init_params = predefined.InitParameters()

    parse_args(init_params)

    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    # init_params.depth_mode = sl.DEPTH_MODE.NEURAL2

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(err)
        exit(1)

    print(f"{init_params=}")
    for k, v in inspect.getmembers(init_params):
        if k.find("__") < 0:
            print(k, v)
    input("hit return key to continue")

    depth_map = sl.Mat()
    point_cloud = sl.Mat()

    runtime_parameters = predefined.RuntimeParameters()
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    runtime_parameters.confidence_threshold = opt.confidence_threshold
    print(f"### {runtime_parameters.confidence_threshold=}")
    for k, v in inspect.getmembers(runtime_parameters):
        if k.find("__") < 0:
            print(k, v)

    fill_modes = [True, False]
    for mode in fill_modes:
        runtime_parameters.enable_fill_mode = mode
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)  # Retrieve depth
            depth_map_img = depth_map.get_data()
            print(f"{depth_map_img.shape=} {depth_map_img.dtype=}" +
            "{all_isfinite(depth_map_img)=} {any_isnan(depth_map_img)=}" +
            "{any_isneginf(depth_map_img)=} {any_isposinf(depth_map_img)=}")
            # 空間座標を得ることが必要。
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            points = point_cloud.get_data()
            print(f"{points.shape=}")
    image.free(memory_type=sl.MEM.CPU)
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
