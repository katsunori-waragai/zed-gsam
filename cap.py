from pathlib import Path

import cv2

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default="captured", help="captured file directory")
parser.add_argument(
    "--is_zed", action="store_true", help="ZED2 stereo camera as USB camera"
)
args = parser.parse_args()

cap = cv2.VideoCapture(0)
outdir = Path(args.outdir)
outdir.mkdir(exist_ok=True, parents=True)
is_zed = args.is_zed

counter = 0
while True:
    r, image = cap.read()
    if is_zed:
        h, w = image.shape[:2]
        image = image[:, : w // 2, :]
    oimg = image.copy()
    cv2.putText(oimg, "s: save", (20, 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0))
    cv2.putText(oimg, "q: quit", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0))
    cv2.imshow("image", oimg)
    key = cv2.waitKey(50)
    if key == ord("q"):
        break
    elif key == ord("s"):
        counter += 1
        oname = outdir / f"cap_{counter:04d}.jpg"
        cv2.imwrite(str(oname), image)
        print(f"saved {oname}")
