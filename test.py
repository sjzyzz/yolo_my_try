import cv2
import yaml
import numpy as np
import argparse
from utils.dataset import create_dataset
from utils.general import check_file, xywh2xyxy
from utils.plot import plot_one_box

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", type=str, default="data/coco128.yaml", help="data.yaml path"
)
opt = parser.parse_args()
opt.data = check_file(opt.data)
with open(opt.data, "r") as f:
    data_dict = yaml.load(f, Loader=yaml.FullLoader)
train_path = data_dict["train"]
names = data_dict["names"]
print(train_path)
dataset, dataloader = create_dataset(train_path, batch_size=4)
# pbar = tqdm(dataloader)
for i, (img, labels) in enumerate(dataloader):
    if i == 0:
        labels = xywh2xyxy(labels, img[0].shape[2], img[0].shape[1])
        for img_id in range(dataloader.batch_size):
            draw_img = img[img_id].permute(1, 2, 0).numpy().astype("uint8")
            draw_img = np.ascontiguousarray(draw_img)
            j = 0
            while j < labels.shape[0]:
                if labels[j][0] == img_id:
                    print(f"{names[int(labels[j][1].item())]}")
                    plot_one_box(
                        labels[j, 2:6], draw_img, label=names[int(labels[j][1].item())],
                    )
                j += 1
            cv2.imshow(
                f"picture {labels[i][0]}", draw_img,
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        break
