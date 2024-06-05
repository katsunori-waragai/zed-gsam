"""
このファイルの中で選択肢をコメントとして表示しよう。
そうすれば、カスタマイズが簡単になる。
"""

import pyzed.sl as sl


def InitParameters():
    """
    datamember の順番はascii 順に記述します。
    """
    init_params = sl.InitParameters()
    # 単位系、座標系の定義は大事
    init_params.coordinate_units = sl.UNIT.METER  # 他の可能性は考えないこと
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP  # 他の可能性は考えないこと

    init_params.camera_fps = 0
    init_params.camera_image_flip = sl.FLIP_MODE.AUTO

    # RESOLUTIONの値が大きいと、複数カメラのときに帯域が足らなくなる。
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    # init_params.camera_resolution = sl.RESOLUTION.HD1200
    # init_params.camera_resolution = sl.RESOLUTION.HD1080
    # init_params.camera_resolution = sl.RESOLUTION.HD720
    # init_params.camera_resolution = sl.RESOLUTION.SVGA
    # init_params.camera_resolution = sl.RESOLUTION.VGA

    # init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    # init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    # init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    init_params.depth_maximum_distance = 50  # [m]
    init_params.depth_minimum_distance = 0.15  # [m] Set the minimum depth perception distance to 15cm

    init_params.depth_stabilization = 1
    init_params.enable_image_enhancement = True
    init_params.enable_image_validity_check = False
    init_params.enable_right_side_measure = True
    init_params.grab_compute_capping_fps = 0
    init_params.open_timeout_sec = 5.0
    init_params.sensors_required = True
    init_params.svo_real_time_mode = False
    return init_params


def RuntimeParameters():
    """
    datamember の順番はascii 順に記述します。
    """
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.enable_depth = True
    # False にすると欠損値を生じます。
    runtime_parameters.enable_fill_mode = False
    # 値を100にすると欠損値が少なくなる方向。値を小さくすると、欠損値が増える。
    runtime_parameters.confidence_threshold = 90  # max = 100
    runtime_parameters.texture_confidence_threshold = 100
    # runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
    runtime_parameters.remove_saturated_areas = True
    return runtime_parameters


def has_datamember():
    """
    old versioned StereoLabs document has different datamembers.
    You can check if such datamembers.
    """
    import inspect

    runtime_parameters = sl.RuntimeParameters()
    obsolute_items = ("sensing_mode",)
    for datamember in obsolute_items:
        keys = [k for k, v in inspect.getmembers(runtime_parameters)]
        print(f"{datamember} {datamember in keys=}")
