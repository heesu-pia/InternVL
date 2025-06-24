import csv
import json
import os
import re

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ===== 설정 =====
FRAMES_PATH = "dataset/에스컬레이터_전도_10"
# OUTPUT_CSV = "autolabel_results.csv"
# OUTPUT_CSV = "autolabel_results_internvl3-2B.csv"
# OUTPUT_CSV = "autolabel_results_internvl3-2B_PrEng.csv"
OUTPUT_JSON = "autolabel_results_internvl3-2B_PrEng2.json"
# MODEL_PATH = "OpenGVLab/InternVL2_5-8B"
MODEL_PATH = "OpenGVLab/InternVL3-2B"
# MODEL_PATH = "OpenGVLab/InternVL3-8B"
IMAGE_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
# PROMPT = (
#     "Is this a situation where someone has fallen on an escalator? "
#     "Estimate the likelihood as a number between 0 and 100. "
#     "Please answer in the exact following format (only numbers, no percent signs):\n"
#     "- Probability of fall: \n"
#     "- Confidence level: \n"
#     "- Reason: "
# )
PROMPT = (
    "You are an expert visual reasoning assistant. Analyze the provided image and classify the person's state on the escalator.\n\n"
    "Classify the scene into one of the following two categories **based on posture and context**:\n\n"
    "**Classification Rules:**\n"
    "- 'fallOnEscalator':\n"
    "    - A person lying on the escalator\n"
    "    - A person sitting on the escalator\n"
    "    - A person kneeling on the escalator (one or both knees touching the surface)\n\n"
    "- 'normal':\n"
    "    - A person standing or walking on the escalator\n"
    "    - A person holding the handrail while in upright posture\n"
    "    - A person bending their knees slightly while still remaining upright (e.g., crouching or squatting **without touching the escalator surface with knees**)\n\n"
    "**Important:**\n"
    "If the posture appears ambiguous or uncertain (e.g., partial crouch, low resolution, unclear knee contact), **prefer to classify as 'fallOnEscalator'** to avoid missing potential fall incidents.\n\n"
    "**Output Requirements:**\n"
    "You must respond in **valid JSON format**, with both fields below:\n"
    '- "category": either "fallOnEscalator" or "normal"\n'
    '- "description": a short explanation of the observed posture (e.g., "person kneeling on the escalator", "person slightly crouching while standing")\n\n'
    "**Example Outputs:**\n"
    "{\n"
    '  "category": "fallOnEscalator",\n'
    '  "description": "person sitting on the escalator"\n'
    "}\n\n"
    "{\n"
    '  "category": "normal",\n'
    '  "description": "person standing upright while holding the handrail"\n'
    "}\n\n"
    "Only provide the JSON object in your response. Do not include any extra text."
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


def extract_json_from_response(response):
    """
    모델 응답에서 JSON 오브젝트만 추출 (코드블록, 불필요한 텍스트 제거)
    """
    # 코드블록(```json ... ```)이 있으면 그 안만 추출
    codeblock = re.search(r"```json(.*?)```", response, re.DOTALL)
    if codeblock:
        response = codeblock.group(1)
    # 중괄호로 감싸진 JSON만 추출
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except Exception:
            pass
    return None


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

    already_labeled = set()
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    already_labeled.add((obj["video_id"], obj["image"]))
                except Exception:
                    continue
    print(f"🔁 Skipping {len(already_labeled)} already-labeled items")

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
                # response에서 category, description 추출 (JSON 파싱)
                result_json = extract_json_from_response(response)
                if (
                    result_json
                    and "category" in result_json
                    and "description" in result_json
                ):
                    category = result_json["category"]
                    description = result_json["description"]
                else:
                    # fallback: description만 텍스트로 저장
                    category = ""
                    description = response.strip()

                # 한 줄씩 JSON으로 저장
                with open(OUTPUT_JSON, mode="a", encoding="utf-8") as f:
                    json.dump(
                        {
                            "video_id": video_id,
                            "image": img_name,
                            "category": category,
                            "description": description,
                        },
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")

                print(f"✅ {video_id}/{img_name} → {category} ({description})")

            except Exception as e:
                print(f"[Error] {img_path}: {e}")
                with open(OUTPUT_JSON, mode="a", encoding="utf-8") as f:
                    json.dump(
                        {
                            "video_id": video_id,
                            "image": img_name,
                            "category": "error",
                            "description": str(e),
                        },
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")


if __name__ == "__main__":
    run_autolabeling()
