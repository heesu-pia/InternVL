import csv
import json

from sklearn.metrics import f1_score, precision_score, recall_score

PRED_JSON = "autolabel_results_internvl3-2B_PrEng.json"  # 예측 결과 (jsonl)
GT_CSV = "autolabel_results_on_review_vf_internvl_5-8b.csv"  # 정답 파일 (video_id, image, label)


def load_labels_from_csv(csv_path):
    labels = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        label_idx = header.index("label") if "label" in header else 5
        vid_idx = header.index("video_id") if "video_id" in header else 1
        img_idx = header.index("image") if "image" in header else 2
        for row in reader:
            key = (row[vid_idx], row[img_idx])
            labels[key] = row[label_idx]
    return labels


def load_labels_from_json(json_path):
    labels = {}
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                key = (obj["video_id"], obj["image"])
                labels[key] = obj["category"]
            except Exception:
                continue
    return labels


def main():
    pred_labels = load_labels_from_json(PRED_JSON)
    gt_labels = load_labels_from_csv(GT_CSV)

    y_true, y_pred = [], []
    for key in gt_labels:
        if key in pred_labels:
            y_true.append(gt_labels[key])
            y_pred.append(pred_labels[key])
        else:
            print(f"예측 결과 없음: {key}")

    valid_labels = {"normal", "fallOnEscalator"}
    filtered = [
        (yt, yp)
        for yt, yp in zip(y_true, y_pred)
        if yt in valid_labels and yp in valid_labels
    ]
    if not filtered:
        print("유효한 라벨이 없습니다.")
        return
    y_true_filtered, y_pred_filtered = zip(*filtered)
    f1 = f1_score(
        y_true_filtered, y_pred_filtered, pos_label="fallOnEscalator", average="binary"
    )
    precision = precision_score(
        y_true_filtered, y_pred_filtered, pos_label="fallOnEscalator", average="binary"
    )
    recall = recall_score(
        y_true_filtered, y_pred_filtered, pos_label="fallOnEscalator", average="binary"
    )

    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
