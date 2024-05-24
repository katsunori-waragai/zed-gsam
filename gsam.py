import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import json
import torch
from PIL import Image

FOLDER_ROOT = Path(__file__).resolve().parent

sys.path.append(str(FOLDER_ROOT / "GroundingDINO"))
sys.path.append(str(FOLDER_ROOT / "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)


# segment anything
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor

COLOR_MAP = {
    0: [0, 0, 0],  # 黒
    1: [0, 255, 0],  # 緑
    2: [0, 0, 255],  # 青
    3: [255, 0, 0],  # 赤
    4: [255, 255, 0],  # 黄色
    5: [255, 0, 255],  # マゼンタ
    6: [0, 255, 255],  # シアン
    7: [128, 128, 128],  # グレー
    8: [128, 0, 0],  # マルーン
    9: [128, 128, 0],  # オリーブ
    10: [0, 128, 0],  # ダークグリーン
    11: [0, 128, 128],  # ティール
    12: [0, 0, 128],  # ネイビー
    13: [255, 165, 0],  # オレンジ
    14: [255, 215, 0],  # ゴールド
    15: [173, 216, 230],  # ライトブルー
    16: [75, 0, 130],  # インディゴ
    17: [240, 128, 128],  # ライトコーラル
    18: [244, 164, 96],  # サドルブラウン
    19: [60, 179, 113],  # ミディアムシーブルー
}


def to_json(label_list: List[str], box_list: List, background_value: int = 0) -> Dict:
    value = background_value
    json_data = [{"value": value, "label": "background"}]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split("(")
        logit = logit[:-1]  # the last is ')'
        json_data.append(
            {
                "value": value,
                "label": name,
                "logit": float(logit),
                "box": box.numpy().tolist(),
            }
        )
    return json_data


def colorize(segmentation_result: np.ndarray) -> np.ndarray:
    height, width = segmentation_result.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    num_colors = len(COLOR_MAP)
    maxint = int(np.max(segmentation_result.flatten()))
    for i in range(maxint + 1):
        color_image[segmentation_result == i] = COLOR_MAP[i % num_colors]
    return color_image


def pil2cv(image: Image) -> np.ndarray:
    """PIL型 -> OpenCV型"""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def cv2pil(image: np.ndarray) -> Image:
    """OpenCV型 -> PIL型"""
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def _load_dino_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def _get_grounding_output(
    dino_model,
    torch_image,
    caption,
    box_threshold,
    text_threshold,
    with_logits=True,
    device="cuda",
):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    dino_model = dino_model.to(device)
    torch_image = torch_image.to(device)
    with torch.no_grad():
        outputs = dino_model(torch_image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = dino_model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def gen_mask_img(mask_list: torch.Tensor, background_value=0) -> torch.Tensor:
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = background_value + idx + 1
    return mask_img


def overlay_image(
    boxes_filt: List,
    pred_phrases: List[str],
    cvimage: np.ndarray,
    colorized: np.ndarray,
    alpha=0.3,
) -> np.ndarray:
    blend_image = np.array(alpha * colorized + (1 - alpha) * cvimage, dtype=np.uint8)
    for box, label in zip(boxes_filt, pred_phrases):
        print(f"{box=} {label=}")
        x1, y1, x2, y2 = [int(a) for a in box]
        cv2.rectangle(blend_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)
        cv2.putText(
            blend_image,
            label,
            (x1, y1),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(255, 0, 255),
            thickness=2,
        )
    return blend_image


def modify_boxes_filter(boxes_filt, W: int, H: int):
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    return boxes_filt

SAM_CHECKPOINT_FILES = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

SAM_HQ_CHECKPOINT_FILES = {
    "vit_h": "sam_hq_vit_h.pth",
    "vit_l": "sam_hq_vit_l.pth",
    "vit_b": "sam_hq_vit_b.pth",
    "vit_tiny": "sam_hq_vit_tiny",
}
print(f"{SAM_CHECKPOINT_FILES['vit_h']=}")
print(f"{SAM_CHECKPOINT_FILES['vit_h'].split('/')[-1]=}")

def name_part(url_filename):
    return url_filename.split("/")[-1]

@dataclass
class GroundedSAMPredictor:
    """
    GroundingDino and Segment Anything

    base
    large
    huge
    """
    # GroundingDino のPredictor
    # SAMのPredictor
    dino_config_file: str = str(
        FOLDER_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    dino_checkpoint: str = str(FOLDER_ROOT / "groundingdino_swint_ogc.pth")
    device: str = "cuda"
    sam_version: str = "vit_l"  # "SAM ViT version: vit_b / vit_l / vit_h"
    use_sam_hq: bool = False
    text_prompt: str = "arm . cup . keyboard . table . plate . bottle . PC . person"
    box_threshold: float = 0.3
    text_threshold: float = 0.25

    def __post_init__(self):
        assert self.sam_version in SAM_CHECKPOINT_FILES
        self.sam_checkpoint: str = str(FOLDER_ROOT / name_part(SAM_CHECKPOINT_FILES[self.sam_version]))  # ex.  "sam_vit_h_4b8939.pth"
        self.sam_hq_checkpoint: str = str(FOLDER_ROOT / SAM_HQ_CHECKPOINT_FILES[self.sam_version])  # ex. "sam_hq_vit_h.pth"
        # 各modelの設定をする。
        self.dino_model = _load_dino_model(
            self.dino_config_file, self.dino_checkpoint, device=self.device
        )
        # initialize SAM
        sam_ckp = self.sam_hq_checkpoint if self.use_sam_hq else self.sam_checkpoint
        if self.use_sam_hq:
            self.sam_predictor = SamPredictor(
                sam_hq_model_registry[self.sam_version](checkpoint=sam_ckp).to(
                    self.device
                )
            )
        else:
            self.sam_predictor = SamPredictor(
                sam_model_registry[self.sam_version](checkpoint=sam_ckp).to(self.device)
            )
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def infer_all(self, cvimage: np.ndarray):
        used = {}
        image_pil = cv2pil(cvimage)
        H, W = cvimage.shape[:2]
        torch_image, _ = self.transform(image_pil, None)  # 3, h, w
        t0 = cv2.getTickCount()
        boxes_filt, pred_phrases = _get_grounding_output(
            self.dino_model,
            torch_image,
            self.text_prompt,
            self.box_threshold,
            self.text_threshold,
            device=self.device,
        )
        boxes_filt = modify_boxes_filter(boxes_filt, W, H)
        t1 = cv2.getTickCount()
        used["dino"] = (t1 - t0) / cv2.getTickFrequency()
        t2 = cv2.getTickCount()
        if pred_phrases:
            self.sam_predictor.set_image(cvimage)
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
                boxes_filt, cvimage.shape[:2]
            ).to(self.device)
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.device),
                multimask_output=False,
            )
        else:
            C = len(pred_phrases)
            masks = torch.from_numpy(np.full((C, H, W), False, dtype=np.bool))

        t3 = cv2.getTickCount()
        used["sam"] = (t3 - t2) / cv2.getTickFrequency()

        # 検出結果はデータメンバーとして保持する。
        self.pred_phrases = pred_phrases
        self.masks = masks
        self.boxes_filt = boxes_filt
        self.colorized = colorize(gen_mask_img(masks).numpy())
        self.used = used


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
        colorized = colorize(gen_mask_img(masks).numpy())
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
