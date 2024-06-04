import numpy as np
import cv2
import skimage
def gen_hsv_image(gray, masks):
    """
    gray scale 画像をBrightness
    maskの値を Hue
    に対応付けた画像を返す。
    ただし、maskの値が０の領域はbackground なので、Saturation=0 にして色を付けない。

    欠点：元のdepthのグレースケール画像で黒いと、heuでマスクの種類を指定しても、
    黒は黒いままなので表示には向かない。
    """
    if len(gray.shape) == 3:
        gray = gray[:, :, 0]

    mask_vals = np.unique(masks)
    maxv = max(mask_vals)
    print(f"{maxv=}")
    print(f"{masks.shape=}")
    hue = np.array(masks.astype(dtype=np.float32) * (255 / maxv), dtype=np.uint8)
    saturation = np.full(gray.shape, 255, dtype=np.uint8)
    saturation[masks == 0] = 0
    value = gray.astype(dtype=np.uint8)
    return cv2.merge((hue, saturation, value))

if __name__ == "__main__":
    import skimage
    img = skimage.data.astronaut()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masks = np.zeros(gray.shape[:2], dtype=np.uint8)
    boxes = [(0.1, 0.1, 0.2, 0.2),
               (0.3, 0.3, 0.4, 0.4),
               (0.4, 0.2, 0.6, 0.5),
             (0.5, 0.5, 0.6, 0.6),
             (0.6, 0.8, 0.7, 0.9),
             ]
    H, W = img.shape[:2]
    for i, (x1f, y1f, x2f, y2f) in enumerate(boxes):
        x1 = int(x1f * W)
        x2 = int(x2f * W)
        y1 = int(y1f * H)
        y2 = int(y2f * H)
        masks[y1:y2, x1:x2] = i + 1

    hsv_image = gen_hsv_image(gray, masks)
    cv2.imshow("hsv", hsv_image)
    cv2.waitKey(-1)
    bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    cv2.imwrite("hsv_view.jpg", bgr)
