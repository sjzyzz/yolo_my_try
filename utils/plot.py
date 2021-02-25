import cv2
import random
import numpy as np
import yaml


def plot_one_box(x, img, label=None, color=None, thickness=None):
    thickness = thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness)
    if label:
        tf = max(thickness - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=thickness / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            thickness / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


if __name__ == "__main__":
    with open("data/coco128.yaml") as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    names = data_dict["names"]
    img = cv2.imread(
        "/Users/yangzizhang/coco128/images/train2017/000000000009.jpg",
        cv2.IMREAD_UNCHANGED,
    )
    h, w = img.shape[:2]
    with open("/Users/yangzizhang/coco128/labels/train2017/000000000009.txt") as f:
        labels = np.array(
            [x.split() for x in f.read().strip().splitlines()], dtype=np.float
        )
    for i in range(labels.shape[0]):
        cat = int(labels[i][0])
        x1 = labels[i][1] - labels[i][3] / 2
        x2 = labels[i][1] + labels[i][3] / 2
        y1 = labels[i][2] - labels[i][4] / 2
        y2 = labels[i][2] + labels[i][4] / 2
        plot_one_box((x1 * w, y1 * h, x2 * w, y2 * h), img, names[cat])
    # cv2.imshow("jordan with his shoes", img)
    cv2.imwrite("9.png", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
