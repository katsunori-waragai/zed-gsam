import inspect
from dataclasses import dataclass


import cv2
import numpy as np
import matplotlib.pyplot as plt


def bbox_to_xyxy(bounding_box_2d, as_int=True):
    """
    4点の座標から、左上、右下の座標に変換する。

    Args:
        bounding_box_2d: ４点の座標
        as_int: Trueのとき、int型で返す。

    Returns:　左上、右下の座標

    """
    xlist = [x for x, _ in bounding_box_2d]
    ylist = [y for _, y in bounding_box_2d]
    xmin = min(xlist)
    xmax = max(xlist)

    ymin = min(ylist)
    ymax = max(ylist)

    if as_int:
        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

    return ((xmin, ymin), (xmax, ymax))


def bbox_to_xyzxyz(bounding_box_3d, as_int=False):
    """
    4点の座標から、左上、右下の座標に変換する。

    Args:
        bounding_box_2d: ４点の座標
        as_int: Trueのとき、int型で返す。

    Returns:　左上、右下の座標

    """
    xlist = [x for x, _, _ in bounding_box_3d]
    ylist = [y for _, y, _ in bounding_box_3d]
    zlist = [z for _, _, z in bounding_box_3d]
    if not (len(xlist) == 8 and len(ylist) == 8 and len(zlist) == 8):
        return None
    xmin = min(xlist)
    xmax = max(xlist)

    ymin = min(ylist)
    ymax = max(ylist)

    zmin = min(zlist)
    zmax = max(zlist)

    if as_int:
        xmin, xmax, ymin, ymax, zmin, zmax = (
            int(xmin),
            int(xmax),
            int(ymin),
            int(ymax),
            int(zmin),
            int(zmax),
        )

    return ((xmin, ymin, zmin), (xmax, ymax, zmax))


def point_selection(xyz_data: np.ndarray, xlim=[-5, 5], ylim=[-5, 5], zlim=[-5, 5]) -> np.ndarray:
    """
    xyz_data[:, 0]: x coordinate
    xyz_data[:, 1]: y coordinate
    xyz_data[:, 2]: z coordinate
    """
    finite_points = []
    for i in range(xyz_data.shape[0]):
        if not np.isfinite(xyz_data[i, :]).all():
            continue
        elif not (xlim[0] <= xyz_data[i, 0] <= xlim[1]):
            continue
        elif not (ylim[0] <= xyz_data[i, 1] <= ylim[1]):
            continue
        elif not (zlim[0] <= xyz_data[i, 2] <= zlim[1]):
            continue
        else:
            finite_points.append(xyz_data[i, :])
    return np.array(finite_points)


def point_selection_by_box(xyz_data, rgba, xlim=[-20, 20], ylim=[-20, 20], zlim=[-20, 20]):
    h, w = xyz_data.shape[:2]
    npoints = h * w
    assert npoints > 0
    point_xyz = np.reshape(xyz_data[:, :, :3], (npoints, 3))
    point_colors = np.reshape(rgba[:, :, :3], (npoints, 3))

    assert point_xyz.shape == point_colors.shape
    indexes = point_indexes_within_box(point_xyz, xlim, ylim, zlim)
    point_xyz = np.take(point_xyz, indexes, axis=0)
    point_colors = np.take(point_colors, indexes, axis=0)
    return point_xyz, point_colors


def point_indexes_within_box(xyzs: np.array, xlim, ylim, zlim):
    xmin, xmax = xlim
    ymin, ymax = ylim
    zmin, zmax = zlim

    indexes = []
    for i in range(xyzs.shape[0]):
        if not (xmin <= xyzs[i, 0] <= xmax):
            continue
        elif not (ymin <= xyzs[i, 1] <= ymax):
            continue
        elif not (zmin <= xyzs[i, 2] <= zmax):
            continue
        if not np.isfinite(xyzs[i, :]).all():
            continue
        else:
            indexes.append(i)
    return indexes


def points_isfinite(xyzs: np.array):
    candidate_points = xyzs.shape[0]
    indexes = [i for i in range(candidate_points) if np.isfinite(xyzs[i, :]).all()]
    return np.take(xyzs, indexes, axis=0)


def mask_by_depth(cv_depth_img: np.ndarray) -> np.ndarray:
    """
    Otsu の２値化で近距離がnonzero であるマスク画像を返す。
    """
    if len(cv_depth_img.shape) > 2:
        cv_depth_img = cv_depth_img[:, :, 0]
    _, th = cv2.threshold(cv_depth_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def gen_transparent_img(cvimg: np.ndarray, mask: np.ndarray, full_transparent=False) -> np.ndarray:
    half_transparent = not full_transparent
    cvimg_with_mask = cvimg.copy()
    if cvimg_with_mask.shape[2] == 3:
        cvimg_with_mask = cv2.cvtColor(cvimg, cv2.COLOR_BGR2BGRA)
    elif cvimg_with_mask.shape[2] == 4:
        cvimg_with_mask = cvimg.copy()
    else:
        print("unsupported input")
    assert cvimg_with_mask.shape[2] == 4
    cvimg_with_mask[:, :, 3] = np.where(mask == 0, 128, mask) if half_transparent else mask
    return cvimg_with_mask


def image_by_otsu(cvimg: np.ndarray, cv_depth_img: np.ndarray) -> np.ndarray:
    mask = mask_by_depth(cv_depth_img)
    return gen_transparent_img(cvimg, mask)


def select_point_cloud_by_img_roi(cvimg: np.ndarray, xyz_data: np.ndarray) -> np.ndarray:
    ROI = cv2.selectROI("select ROI", cvimg, fromCenter=False, showCrosshair=False)
    xl = ROI[0]
    yu = ROI[1]
    xr = ROI[0] + ROI[2]
    yd = ROI[1] + ROI[3]
    return xyz_data[yu:yd, xl:xr, :]


def point_selection_by_mask(xyz_data: np.ndarray, rgba: np.ndarray, mask: np.ndarray):
    assert xyz_data.shape[:2] == rgba.shape[:2]
    assert xyz_data.shape[:2] == mask.shape[:2]
    h, w = xyz_data.shape[:2]
    npoints = h * w
    assert npoints > 0
    assert rgba.shape[2] == 4
    point_xyz = np.reshape(xyz_data[:, :, :3], (npoints, 3))
    point_colors = np.reshape(rgba[:, :, :3], (npoints, 3))
    point_mask = np.reshape(mask, (npoints))
    assert point_xyz.shape == point_colors.shape
    indexes = [i for i in range(len(point_mask)) if point_mask[i]]
    point_xyz = np.take(point_xyz, indexes, axis=0)
    point_colors = np.take(point_colors, indexes, axis=0)
    return point_xyz, point_colors


def view_xyz(xyz_data: np.ndarray, rgba: np.ndarray):
    """
    xyz_data: 入力画像の[i,j] に対応する空間座標の値
    xyz_data[:, 0]: x for the image
    xyz_data[:, 1]: y for the image
    xyz_data[:, 2]: y for the image
    """
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.imshow(xyz_data[:, :, 2], vmax=0.0)
    plt.colorbar()
    plt.title("z")
    plt.subplot(2, 2, 2)
    plt.imshow(xyz_data[:, :, 0])
    plt.colorbar()
    plt.title("x")
    plt.subplot(2, 2, 3)
    plt.imshow(xyz_data[:, :, 1])
    plt.colorbar()
    plt.title("y")
    plt.subplot(2, 2, 4)
    plt.imshow(rgba)
    plt.pause(0.01)


def is_primitive(obj) -> bool:
    return (
        isinstance(obj, bool)
        or isinstance(obj, int)
        or isinstance(obj, float)
        or isinstance(obj, complex)
        or isinstance(obj, list)
        or isinstance(obj, tuple)
        or isinstance(obj, str)
        or isinstance(obj, str)
        or isinstance(obj, set)
    )


def show_params(some_params, show_all=False):
    if is_primitive(some_params):
        print(f"{some_params}")
    elif isinstance(some_params, np.ndarray):
        print(f"{some_params.shape} {some_params.dtype}")
        print(some_params)
    else:
        print(f"{str(some_params)}")
        for k, v in inspect.getmembers(some_params):
            if show_all or k.find("__") < 0:
                print(f"{type(some_params)} {k} {v}")


@dataclass
class TimeStampQueue:
    timestamps = []

    def put(self, val):
        self.timestamps.append(val)
        while len(self.timestamps) > 2:
            self.timestamps.pop(0)

    def diff(self):
        if len(self.timestamps) == 2:
            return self.timestamps[1] - self.timestamps[0]
        else:
            return None

    def __len__(self):
        return len(self.timestamps)


def plot_xyz(point_xyz: np.ndarray, imgname=None):
    """
    point_xyz 点群の空間座標xyz
    point_xyz[:, 0]: x座標
    point_xyz[:, 1]: y座標
    point_xyz[:, 2]: z座標
    """
    plt.figure(2)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(point_xyz[:, 0], point_xyz[:, 1], ".", label="x-y(vertical)")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y (vertical)")
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(point_xyz[:, 2], point_xyz[:, 1], ".", label="z(depth)-y(vertical)")
    plt.legend()
    plt.xlabel("z (depth)")
    plt.ylabel("y (vertical)")
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(point_xyz[:, 0], point_xyz[:, 2], ".", label="x-z(depth)")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("z (depth)")
    plt.grid(True)
    if imgname:
        plt.savefig(imgname)
    plt.pause(1)


def plot_xyz_for_image_array(xyz_data: np.ndarray, imgname=None):
    """
    xyz_data: 入力画像の[i,j] に対応する空間座標の値
    xyz_data[:, 0]: x for the image
    xyz_data[:, 1]: y for the image
    xyz_data[:, 2]: y for the image

    """
    xs = np.array(xyz_data[:, :, 0]).flatten()
    ys = np.array(xyz_data[:, :, 1]).flatten()
    zs = np.array(xyz_data[:, :, 2]).flatten()
    plt.figure(3)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(xs, ys, ".")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(ys, zs, ".")
    plt.xlabel("y")
    plt.ylabel("z")
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(xs, zs, ".")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.grid(True)
    if imgname:
        plt.savefig(imgname)
    plt.pause(1)
