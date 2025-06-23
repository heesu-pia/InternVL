import csv
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ===== 설정 =====
FRAMES_PATH = "dataset/에스컬레이터_전도_10"
# OUTPUT_CSV = "autolabel_results.csv"
OUTPUT_CSV = "autolabel_results_internvl3-2B.csv"
# MODEL_PATH = "OpenGVLab/InternVL2_5-8B"
MODEL_PATH = "OpenGVLab/InternVL3-2B"
IMAGE_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
PROMPT = (
    "<image>\n"
    "Is this a situation where someone has fallen on an escalator? "
    "Estimate the likelihood as a number between 0 and 100 and describe your confidence. "
    "Format:\n"
    "- Probability of fall: [0–100]\n"
    "- Confidence level: [0–100]\n"
    "- Reason: [your explanation]"
)
# ==================


def build_transform():
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def preprocess_image(img):
    transform = build_transform()
    return transform(img).unsqueeze(0)


def parse_probability_confidence_reason(response: str):
    prob, conf = -1, -1
    reason_lines = []
    is_reason = False

    for line in response.splitlines():
        if "Probability of fall" in line:
            try:
                prob = int(line.split(":")[1].strip().replace("%", "").split()[0])
            except:
                pass
        elif "Confidence level" in line:
            try:
                conf = int(line.split(":")[1].strip().replace("%", "").split()[0])
            except:
                pass
        elif "Reason:" in line:
            is_reason = True
            reason_lines.append(line.split("Reason:")[1].strip())
        elif is_reason:
            reason_lines.append(line.strip())

    reason = " ".join(reason_lines)
    return prob, conf, reason


def classify_label(prob: int, conf: int) -> str:
    if prob >= 50 and conf >= 50:
        return "fallOnEscalator"
    elif 0 <= prob <= 100 and 0 <= conf <= 100:
        return "normal"
    return "error"


def load_already_labeled(csv_path):
    labeled = set()
    if not os.path.exists(csv_path):
        return labeled

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                labeled.add((row[0], row[1]))  # (video_id, image)
    return labeled


def run_autolabeling():
    print("🚀 Loading InternVL model...")
    model = (
        AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .to(DEVICE)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=False
    )

    # 🔁 Resume support
    already_labeled = load_already_labeled(OUTPUT_CSV)
    print(f"🔁 Skipping {len(already_labeled)} already-labeled items")

    # 최초 실행 시 헤더 작성
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "video_id",
                    "image",
                    "fall_probability",
                    "confidence",
                    "label",
                    "reason",
                ]
            )

    for video_id in tqdm(sorted(os.listdir(FRAMES_PATH))):
        folder_path = os.path.join(FRAMES_PATH, video_id)
        if not os.path.isdir(folder_path):
            continue

        for img_name in sorted(os.listdir(folder_path)):
            if not img_name.lower().endswith(".jpg"):
                continue

            if (video_id, img_name) in already_labeled:
                continue

            img_path = os.path.join(folder_path, img_name)
            try:
                pil_img = Image.open(img_path).convert("RGB")
                pixel_values = preprocess_image(pil_img).to(DEVICE, dtype=DTYPE)
                response = model.chat(
                    tokenizer, pixel_values, PROMPT, {"max_new_tokens": 512}
                )
                prob, conf, reason = parse_probability_confidence_reason(response)
                label = classify_label(prob, conf)

                # 한 줄 저장
                with open(OUTPUT_CSV, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([video_id, img_name, prob, conf, label, reason])

                print(f"✅ {video_id}/{img_name} → {label} (prob={prob}, conf={conf})")

            except Exception as e:
                print(f"[Error] {img_path}: {e}")
                with open(OUTPUT_CSV, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([video_id, img_name, -1, -1, "error", ""])


if __name__ == "__main__":
    run_autolabeling()
