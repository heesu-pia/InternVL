# internvl_video_inference.py
import os

import cv2
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# ===== Configuration =====
FRAMES_PATH = "dataset/에스컬레이터_전도_10"
OUTPUT_PATH = "autolabel_results"
MODEL_PATH = "OpenGVLab/InternVL2_5-8B"
FRAME_SKIP = 30
IMAGE_SIZE = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
# ==========================

PROMPT = (
    "<image>\n"
    "Is this a situation where someone has fallen on an escalator? "
    "Estimate the likelihood as a number between 0 and 100 and describe your confidence. "
    "Format:\n"
    "- Probability of fall: [0–100]\n"
    "- Confidence level: [0–100]\n"
    "- Reason: [your explanation]"
)


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


def parse_value(line):
    try:
        value_str = line.split(":")[1].strip().replace("%", "").split()[0]
        return int(value_str)
    except:
        return None


def draw_wrapped_text(img, text, origin, font, font_scale, color, thickness, max_width):
    x, y = origin
    words = text.split()
    lines, line = [], ""

    for word in words:
        test_line = line + word + " "
        text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        if text_size[0] > max_width:
            lines.append(line)
            line = word + " "
        else:
            line = test_line
    lines.append(line)

    for i, line in enumerate(lines):
        y_pos = y + i * int(30 * font_scale)
        cv2.putText(img, line.strip(), (x, y_pos), font, font_scale, color, thickness)


def draw_overlay(frame, result_text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 255, 0)
    margin = 30
    max_width = frame.shape[1] - 2 * margin

    lines = result_text.strip().split("\n")
    top_lines = lines[:2]
    reason_text = "\n".join(lines[2:])
    y_start = 30

    for i, line in enumerate(top_lines):
        y = y_start + i * int(30 * font_scale)
        cv2.putText(frame, line, (margin, y), font, font_scale, color, thickness)

    draw_wrapped_text(
        frame,
        reason_text,
        (margin, y_start + len(top_lines) * 30),
        font,
        font_scale,
        color,
        thickness,
        max_width,
    )

    try:
        prob = parse_value(top_lines[0])
        conf = parse_value(top_lines[1])
        if prob is not None and conf is not None and prob > 50 and conf > 50:
            radius = 20
            center = (frame.shape[1] - radius - 20, radius + 20)
            cv2.circle(frame, center, radius, (0, 0, 255), thickness=-1)
    except Exception as e:
        print(f"[Warning] Failed to parse values: {e}")

    return frame


def run_internvl_inference():
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

    vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    h, w = vr[0].shape[0], vr[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
    total_frames = len(vr)
    results = {}

    for i in range(0, total_frames, FRAME_SKIP):
        pil_img = Image.fromarray(vr[i].asnumpy()).convert("RGB")
        pixel_values = preprocess_image(pil_img).to(DEVICE, dtype=DTYPE)

        gen_cfg = dict(max_new_tokens=512, do_sample=False)
        try:
            response = model.chat(tokenizer, pixel_values, PROMPT, gen_cfg)
        except Exception as e:
            print(f"[Error @ frame {i}] {e}")
            response = "Prediction failed"

        results[i] = response
        print(
            f"✅ Frame {i}: {response.splitlines()[0] if isinstance(response, str) else response}"
        )

    for i in range(total_frames):
        frame = vr[i].asnumpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        anchor = (i // FRAME_SKIP) * FRAME_SKIP
        if anchor in results:
            frame = draw_overlay(frame, results[anchor])
        writer.write(frame)

    writer.release()
    print(f"🎬 Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_internvl_inference()
