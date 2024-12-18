"""
depth画像を見るサンプルスクリプト

"""

import pyzed.sl as sl
import argparse

import cv2

import zedhelper.handmark
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


def main():
    # Create a Camera object
    zed = sl.Camera()

    use_hand = True

    if use_hand:
        hand_marker = zedhelper.handmark.HandMarker()

    # Create a InitParameters object and set configuration parameters
    init_params = predefined.InitParameters()

    parse_args(init_params)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(err)
        exit(1)

    # Enable object detection module
    camera_info = zed.get_camera_information()
    # Create OpenGL viewer

    # Configure object detection runtime parameters

    # Create ZED objects filled in the main loop
    image = sl.Mat()
    depth_map = sl.Mat()
    depth_for_display = sl.Mat()

    # Set runtime parameters
    runtime_parameters = predefined.RuntimeParameters()
    for k, v in inspect.getmembers(runtime_parameters):
        if k.find("__") < 0:
            print(k, v)

    runtime_parameters.enable_fill_mode = True
    while True:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)  # Retrieve depth
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)  # near to camera is white
            # Retrieve objects
            cvimg = image.get_data()
            cv_depth_img = depth_for_display.get_data()

            if use_hand:
                detection_result = hand_marker.detect(cvimg)
                annotated_image = hand_marker.draw_landmarks(detection_result)
                cv2.imshow("annotated_image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            cv2.imshow("depth_for_display", cv_depth_img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    image.free(memory_type=sl.MEM.CPU)
    # Disable modules and close camera

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
    main()
