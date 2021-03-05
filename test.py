# import cv2
# import yaml
# import numpy as np
# import argparse
# from utils.dataset import create_dataset, LoadImages
# from utils.general import check_file, xywh2xyxy
# from utils.plot import plot_one_box
# from models.yolo import Model

# # parser = argparse.ArgumentParser()
# # parser.add_argument(
# #     "--data", type=str, default="data/coco128.yaml", help="data.yaml path"
# # )
# # opt = parser.parse_args()
# # opt.data = check_file(opt.data)
# # with open(opt.data, "r") as f:
# #     data_dict = yaml.load(f, Loader=yaml.FullLoader)
# # train_path = data_dict["train"]
# # names = data_dict["names"]
# # print(train_path)
# # dataset, dataloader = create_dataset(train_path, batch_size=4)

# # model = Model()

# # # pbar = tqdm(dataloader)
# # for i, (img, labels) in enumerate(dataloader):
# #     if i == 0:
# #         pred = model(img)
# #         for p in pred:
# #             print(p.shape)
# #     else:
# #         break
# images = LoadImages("data/images")
# for i, img in enumerate(images):
#     cv2.show(str(i), img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
import torch

# a = torch.tensor([[[1, 2, 2], [1, 2, 3]], [[1, 2, 2], [1, 2, 3]]])
# print(a[..., 2] > 2)
# print(a[a[..., 2] > 2])
def assign(a):
    a[2] = 0


a = torch.tensor([0, 1, 2])
assign(a)
print(a)
e = a[1]
e = 5
print(a)
