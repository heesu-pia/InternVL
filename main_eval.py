import csv

from sklearn.metrics import f1_score

PRED_CSV = "autolabel_results.csv"  # InternVL 예측 결과
GT_CSV = "autolabel_results_on_review_vf.csv"  # 정답 파일 (video_id, image, label)


def load_labels_from_csv(csv_path):
    labels = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        label_idx = header.index("label") if "label" in header else 5
        vid_idx = header.index("video_id") if "video_id" in header else 1
        img_idx = header.index("image") if "image" in header else 2
        for row in reader:
            key = (row[vid_idx], row[img_idx])
            labels[key] = row[label_idx]
    return labels


def main():
    pred_labels = load_labels_from_csv(PRED_CSV)
    gt_labels = load_labels_from_csv(GT_CSV)

    y_true, y_pred = [], []
    for key in gt_labels:
        if key in pred_labels:
            y_true.append(gt_labels[key])
            y_pred.append(pred_labels[key])
        else:
            print(f"예측 결과 없음: {key}")

    f1 = f1_score(y_true, y_pred, pos_label="fallOnEscalator", average="binary")
    print(f"F1-score: {f1:.4f}")


if __name__ == "__main__":
    main()
