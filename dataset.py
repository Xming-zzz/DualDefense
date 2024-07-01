import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from torchvision.transforms import transforms


class CustomDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
        # print(f'len : {len(self.ids)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image = Image.open(self.ids[idx])
        image = Image.fromarray(cv2.imread(self.images[idx]))
        # image = cv2.imread(self.ids[idx])

        if self.transform:
            image = self.transform(image)

        return image


def split_dataset(path, train_transform=None, test_transform=None, val_ratio=0.2, test_ratio=0.2):
    images = [x.path for x in os.scandir(path) if x.name.endswith(".jpg") or x.name.endswith(".png")]
    # images = sorted(images)
    total_len = len(images)
    train_images = images[: int(total_len * (1 - val_ratio - test_ratio))]
    val_images = images[int(total_len * (1 - val_ratio - test_ratio)): int(total_len * (1 - test_ratio))]
    test_images = images[int(total_len * (1 - test_ratio)):]
    return CustomDataset(train_images, train_transform), CustomDataset(val_images, test_transform), CustomDataset(
        test_images, test_transform)


'''
trans_test = transforms.Compose([
    transforms.ToTensor()
])

background = cv2.imread('')
cv2.imwrite('tmp1.png', background)
yuv_background = cv2.cvtColor(background, cv2.COLOR_BGR2YUV)  # 
Y, U, V = yuv_background[..., 0], yuv_background[..., 1], yuv_background[..., 2]
YY = trans_test(Y)
YYY = YY.permute(1, 2, 0).squeeze().numpy() * 255
print(yuv_background[..., 0] == YYY)
x = yuv_background[..., 0] == YYY

yuv_background[..., 0] = YYY
mm = cv2.cvtColor(yuv_background, cv2.COLOR_YUV2BGR)
cv2.imwrite('tmp2.png', mm)

'''
