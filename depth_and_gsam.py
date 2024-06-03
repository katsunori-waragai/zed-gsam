"""
depth画像を見るサンプルスクリプト

"""

import pyzed.sl as sl
import argparse

import cv2
import numpy as np

import zedhelper.handmark
from zedhelper import predefined

import gsam_module

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


def main():
    prompt = "bottle . person . box"
    watching_obj = "box"
    gsam_predictor = gsam_module.GroundedSAMPredictor(
        text_prompt=prompt,
        text_threshold=0.25,
        box_threshold=0.3,
        use_sam_hq=False,
    )

    # Create a Camera object
    zed = sl.Camera()

    use_hand = True

    if use_hand:
        hand_marker = zedhelper.handmark.HandMarker()

    # Create a InitParameters object and set configuration parameters
    init_params = predefined.InitParameters()

    parse_args(init_params)

    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    # init_params.depth_mode = sl.DEPTH_MODE.NEURAL2

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(err)
        exit(1)

    print(f"{init_params=}")
    for k, v in inspect.getmembers(init_params):
        if k.find("__") < 0:
            print(k, v)
    input("hit return key to continue")

    # Enable object detection module
    camera_info = zed.get_camera_information()
    # Create OpenGL viewer

    # Configure object detection runtime parameters

    # Create ZED objects filled in the main loop
    image = sl.Mat()
    depth_map = sl.Mat()
    depth_for_display = sl.Mat()
    point_cloud = sl.Mat()

    # Set runtime parameters
    runtime_parameters = predefined.RuntimeParameters()
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    # runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
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
            depth_map_img = depth_map.get_data()
            cvimg = image.get_data()
            cv_depth_img = depth_for_display.get_data()

            # 空間座標を得ることが必要。
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            points = point_cloud.get_data()
            print(f"{points.shape=}")
            # points[y, x]で、元画像上の点と対応がつくのかどうか？
            if cvimg is not None:
                print(f"{cvimg.shape=}")
                cvimg_bgr = cvimg[:, :, :3].copy()
                gsam_predictor.infer_all(cvimg_bgr)
                masks = gsam_predictor.masks
                colorized = gsam_module.colorize_torch(gsam_module.gen_mask_img(masks)).cpu().numpy()
                uint_masks = gsam_module.gen_mask_img(masks).cpu().numpy()
                mask_val = np.unique(uint_masks).astype(np.int16)
                # mask_val が連続的な整数ではないことが判明した。
                print(f"{mask_val=}　{len(mask_val)}")

                pred_phrases = gsam_predictor.pred_phrases
                boxes_filt = gsam_predictor.boxes_filt
                blend_image = gsam_module.overlay_image(boxes_filt, pred_phrases, cvimg_bgr, colorized)
                blend_image = resize_image(blend_image, 0.5)
                C, H, W = uint_masks.shape[:3]
                assert C == 1
                selected_list = points_by_segmentation(points, uint_masks.reshape(H, W))
                print(f"{len(pred_phrases)=}")
                print(f"{len(selected_list)=}")
                # assert len(pred_phrases) == len(selected_list)

                import matplotlib.pylab as plt
                print("try matplotlib")
                plt.figure(figsize=(10, 6))

                # ax = plt.subplot(2, 2, 1)
                # ax.set_aspect("equal")

                PERCENT_LIMIT = 5
                for i, (selected, phrase) in enumerate(zip(selected_list, pred_phrases)):
                    if phrase.find(watching_obj) > -1:
                        print(f"{i=} {pred_phrases[i]=} {selected=} {phrase=}")
                        print(f"{selected.shape=}")
                        x_per = np.nanpercentile(selected[:, 0], (PERCENT_LIMIT, 100 - PERCENT_LIMIT))
                        y_per = np.nanpercentile(selected[:, 1], (PERCENT_LIMIT, 100 - PERCENT_LIMIT))
                        z_per = np.nanpercentile(selected[:, 2], (PERCENT_LIMIT, 100 - 3 * PERCENT_LIMIT))
                        print(f"{x_per=}")
                        print(f"{y_per=}")
                        print(f"{z_per=}")
                        print(f"{x_per[1] - x_per[0]=}")
                        print(f"{y_per[1] - y_per[0]=}")
                        print(f"{z_per[1] - z_per[0]=}")

                        # plt.plot(selected[:, 0], selected[:, 1], ".")
                # cv2.imshow("output", blend_image)


                # plt.grid(True)
                # plt.xlabel("x [m]")
                # plt.ylabel("y [m]")

                ax1 = plt.subplot(2, 3, 1)
                ax1.set_aspect("equal")
                for i, (selected, phrase) in enumerate(zip(selected_list, pred_phrases)):
                    if phrase.find(watching_obj) > -1:
                        x = selected[:, 0]
                        y = selected[:, 1]
                        z = -selected[:, 2]
                        sc = plt.scatter(x, y, c=z, marker=".", cmap='jet')

                plt.colorbar(sc, label='Z Value')
                plt.xlabel("x [m]")
                plt.ylabel("y [m]")
                plt.grid(True)
                plt.show()

                ax2 = plt.subplot(2, 3, 2)
                ax2.set_aspect("equal")
                for i, (selected, phrase) in enumerate(zip(selected_list, pred_phrases)):
                    if phrase.find(watching_obj) > -1:
                        x = selected[:, 0]
                        y = selected[:, 1]
                        z = -selected[:, 2]
                        sc = plt.scatter(z, y, c=x, marker=".", cmap='jet')

                plt.colorbar(sc, label='x Value')
                plt.xlabel("z [m]")
                plt.ylabel("y [m]")
                plt.grid(True)
                plt.show()

                plt.subplot(2, 3, 3)
                import skimage
                import matplotlib
                is_picked = np.array(255 * uint_masks.reshape(H, W) > 0, dtype=np.uint8)
                is_picked_color = cv2.cvtColor(is_picked, cv2.COLOR_GRAY2RGB)
                print(f"{cv_depth_img.shape=}")
                print(f"{is_picked_color.shape=}")
                # pseudo_color_depth = skimage.color.gray2rgba(cv_depth_img[:, :, 0], alpha=is_picked)
                # pseudo_color_depth = skimage.color.gray2rgba(cv_depth_img[:, :, 0])
                pseudo_color_depth = matplotlib.cm.jet(cv_depth_img[:, :, 0])
                print(f"{pseudo_color_depth.shape=}")
                print(f"{pseudo_color_depth.dtype=}")

                # plt.imshow(is_picked)
                plt.imshow(pseudo_color_depth)
                plt.show()

                plt.subplot(2, 3, 4)
                plt.imshow(colorized)
                plt.show()
                plt.subplot(2, 3, 5)
                plt.imshow(np.abs(depth_map_img), vmin=0.0, vmax=2.0, cmap="jet")
                plt.colorbar()
                plt.subplot(2, 3, 6)
                # colorized と cv_depth_imgとを重ね書きする。
                alpha = 0.2
                blend_image = np.array(alpha * colorized + (1 - alpha) * cvimg[:, :, :3], dtype=np.uint8)
                plt.imshow(blend_image)
                plots_name = "plot_bottle.png"
                plt.savefig(plots_name)
                print(f"saved {plots_name}")


            if use_hand:
                detection_result = hand_marker.detect(cvimg)
                annotated_image = hand_marker.draw_landmarks(detection_result)
                cv2.imshow("annotated_image", resize_image(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR), 0.5))
            cv2.imshow("depth_for_display", resize_image(cv_depth_img, 0.5))
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
