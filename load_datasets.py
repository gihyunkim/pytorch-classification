from torch.utils.data import Dataset
import pickle
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

class CifarDataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        self.transform = transform
        self.x = data[b'data']
        self.y = data[b'fine_labels']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        label = self.y[index]
        r = self.x[index, :1024].reshape(32, 32)
        g = self.x[index, 1024:2048].reshape(32, 32)
        b = self.x[index, 2048:].reshape(32, 32)
        image = np.dstack((r, g, b))
        pil_img = Image.fromarray(image)

        if self.transform:
            image = self.transform(pil_img)
        return image, label

if __name__ == "__main__":
    ds = CifarTrainDataset("./datasets/cifar100/train")
    ds.__getitem__(0)