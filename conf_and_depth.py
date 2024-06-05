"""
depthデータでのNaN, neginf, posinf の出力について確認する。
欠損値に影響するのは、以下のパラメータ
runtime_parameters.enable_fill_mode = True or False
runtime_parameters.confidence_threshold = 0 ~ 100 の数値
runtime_parameters.texture_confidence_threshold = 数値 100
runtime_parameters.remove_saturated_areas = True

今回は、以下の値を固定値とした。
init_params.enable_right_side_measure = True

結論：
- depth_map data にはNaN,　posinfが含まれる。(neginf　は未確認）
- 以下の方法によるpoints_color もNaN,　posinfが含まれる。(neginf　は未確認）
```
zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
points = point_cloud.get_data()
points_color = points[:, :, 3]　
```

confidence_thresholdを下げるにつれて、
depth_map data, points_color とも欠損値がisnan が増えていく。
"""

import argparse
import inspect

import cv2
import numpy as np
import pyzed.sl as sl

from zedhelper import predefined

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
    runtime_parameters.remove_saturated_areas = True
    print(f"### {runtime_parameters.confidence_threshold=}")
    for k, v in inspect.getmembers(runtime_parameters):
        if k.find("__") < 0:
            print(k, v)

    fill_modes = [True, False]
    for mode in fill_modes:
        for conf in [50, 60, 70, 80, 90, 100]:
            print("##################")
            runtime_parameters.enable_fill_mode = mode
            runtime_parameters.confidence_threshold = conf
            for _ in range(10):
                zed.grab(runtime_parameters)
                zed.grab(runtime_parameters)

            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)  # Retrieve depth
                depth_map_data = depth_map.get_data()
                H, W = depth_map_data.shape[:2]
                count_isfinite = np.count_nonzero(np.isfinite(depth_map_data))
                count_isnan = np.count_nonzero(np.isnan(depth_map_data))
                count_isneginf = np.count_nonzero(np.isneginf(depth_map_data))
                count_isposinf = np.count_nonzero(np.isposinf(depth_map_data))
                print(f"""
{runtime_parameters.confidence_threshold=}
{runtime_parameters.enable_fill_mode=}
{depth_map_data.shape=} {depth_map_data.dtype=} %
{count_isfinite=} {100 * count_isfinite / (W * H):.3f} %
{count_isnan=} {100 * count_isnan / (W * H):.3f} %
{count_isneginf=} {100 * count_isneginf / (W * H):.3f} %
{count_isposinf=} {100 * count_isposinf / (W * H):.3f} %
""")
                # 空間座標を得ることが必要。
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                points = point_cloud.get_data()
                print(f"{points.shape=}")
                assert depth_map_data.shape == points.shape[:2]
                points_color = points[:, :, 3]
                count_isfinite_points = np.count_nonzero(np.isfinite(points_color))
                count_isnan_points = np.count_nonzero(np.isnan(points_color))
                count_isneginf_points = np.count_nonzero(np.isneginf(points_color))
                count_isposinf_points = np.count_nonzero(np.isposinf(points_color))
                print(f"""
{count_isfinite_points=} {100 * count_isfinite_points / (W * H): .3f} %
{count_isnan_points=} {100 * count_isnan_points / (W * H): .3f} %
{count_isneginf_points=} {100 * count_isneginf_points / (W * H): .3f} %
{count_isposinf_points=} {100 * count_isposinf_points / (W * H): .3f} %
""")
                points_z = points[:, :, 2]
                count_isfinite_points_z = np.count_nonzero(np.isfinite(points_z))
                count_isnan_points_z = np.count_nonzero(np.isnan(points_z))
                count_isneginf_points_z = np.count_nonzero(np.isneginf(points_z))
                count_isposinf_points_z = np.count_nonzero(np.isposinf(points_z))
                print(f"""
{count_isfinite_points_z=} {100 * count_isfinite_points_z / (W * H): .3f} %
{count_isnan_points_z=} {100 * count_isnan_points_z / (W * H): .3f} %
{count_isneginf_points_z=} {100 * count_isneginf_points_z / (W * H): .3f} %
{count_isposinf_points_z=} {100 * count_isposinf_points_z / (W * H): .3f} %
""")
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
    opt = parser.parse_args()
    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main(opt)
