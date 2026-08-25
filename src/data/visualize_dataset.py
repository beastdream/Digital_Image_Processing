import os
import random
import cv2

def visualize_dataset_samples():
    base_dir = r"d:\Digital_Image_Processing"
    processed_dir = os.path.join(base_dir, "data", "processed", "detection")
    output_base_dir = os.path.join(base_dir, "results", "visualizations")

    class_names = {0: "pothole", 1: "crack", 2: "manhole"}
    colors = {
        0: (0, 0, 255),    # Red for Pothole
        1: (255, 0, 0),    # Blue for Crack
        2: (0, 255, 0)     # Green for Manhole
    }

    splits = ['train', 'val', 'test']

    for split in splits:
        img_dir = os.path.join(processed_dir, "images", split)
        lbl_dir = os.path.join(processed_dir, "labels", split)
        out_dir = os.path.join(output_base_dir, split)

        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(img_dir):
            print(f"Directory not found: {img_dir}")
            continue

        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if len(images) < 10:
            selected_images = images
        else:
            random.seed(42 + len(split))
            selected_images = random.sample(images, 10)

        for fname in selected_images:
            img_path = os.path.join(img_dir, fname)
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")

            img = cv2.imread(img_path)
            if img is None:
                continue

            h_img, w_img, _ = img.shape

            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]

                for line in lines:
                    parts = line.split()
                    if len(parts) != 5:
                        continue

                    cls_id = int(parts[0])
                    xc = float(parts[1]) * w_img
                    yc = float(parts[2]) * h_img
                    w_box = float(parts[3]) * w_img
                    h_box = float(parts[4]) * h_img

                    x1 = int(max(0, xc - w_box / 2))
                    y1 = int(max(0, yc - h_box / 2))
                    x2 = int(min(w_img - 1, xc + w_box / 2))
                    y2 = int(min(h_img - 1, yc + h_box / 2))

                    cname = class_names.get(cls_id, str(cls_id))
                    color = colors.get(cls_id, (255, 255, 255))

                    # Draw Bounding Box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    # Label text
                    label_str = f"[{cls_id}] {cname}"
                    (text_w, text_h), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    # Text background
                    cv2.rectangle(img, (x1, max(0, y1 - text_h - 4)), (x1 + text_w + 4, y1), color, -1)
                    cv2.putText(img, label_str, (x1 + 2, max(text_h, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            out_path = os.path.join(out_dir, f"vis_{fname}")
            cv2.imwrite(out_path, img)

        print(f"Saved {len(selected_images)} visualizations for {split} split to {out_dir}")

if __name__ == "__main__":
    visualize_dataset_samples()
