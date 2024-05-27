import argparse
from pathlib import Path

import cv2
import json

from gsam_module import FOLDER_ROOT, to_json, colorize, colorize_torch, gen_mask_img, overlay_image, \
    SAM_CHECKPOINT_FILES, \
    GroundedSAMPredictor


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Grounded-Segment-Anything")
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="path to image file"
    )
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="outputs",
        required=True,
        help="output directory",
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.3, help="box threshold"
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25, help="text threshold"
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    gsam_predictor = GroundedSAMPredictor(
        text_prompt=args.text_prompt,
        text_threshold=args.text_threshold,
        box_threshold=args.box_threshold,
        use_sam_hq=args.use_sam_hq,
    )

    image_path_list = list(Path(image_dir).glob("*.jpg")) + list(
        Path(image_dir).glob("*.png")
    )
    for p in image_path_list:
        print(p)

    for image_path in sorted(image_path_list):
        cvimage = cv2.imread(str(image_path))
        gsam_predictor.infer_all(cvimage)

        image_path_stem = image_path.stem.replace(" ", "_")
        cv2.imwrite(str(output_dir / f"{image_path_stem}_raw.jpg"), cvimage)

        used_time = gsam_predictor.used.copy()

        masks = gsam_predictor.masks

        t6 = cv2.getTickCount()
        # colorized = colorize(gen_mask_img(masks).cpu().numpy())
        colorized = colorize_torch(gen_mask_img(masks)).cpu().numpy()
        output_mask_jpg = output_dir / f"{image_path_stem}_mask.jpg"
        cv2.imwrite(str(output_mask_jpg), colorized)
        mask_json = output_mask_jpg.with_suffix(".json")
        pred_phrases = gsam_predictor.pred_phrases
        boxes_filt = gsam_predictor.boxes_filt
        with mask_json.open("wt") as f:
            json.dump(to_json(pred_phrases, boxes_filt), f)
        t7 = cv2.getTickCount()
        used_time["save_mask"] = (t7 - t6) / cv2.getTickFrequency()

        t10 = cv2.getTickCount()
        blend_image = overlay_image(boxes_filt, pred_phrases, cvimage, colorized)
        cv2.imwrite(str(output_dir / f"{image_path_stem}_sam.jpg"), blend_image)
        t11 = cv2.getTickCount()
        used_time["save_sam"] = (t11 - t10) / cv2.getTickFrequency()

        print(f"{used_time=}")
        cv2.imshow("output", blend_image)
        key = cv2.waitKey(10)
