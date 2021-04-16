import torch
import numpy as np
import pickle
from torch.utils.data import Dataset



class Wireframe(Dataset):
    def __init__(self, train_list, dataset_dir):
        self.dir = dataset_dir + "wireframe/fisheye_pointlines/"
        self.train_data = np.loadtxt(train_list, dtype=str, delimiter="\n")
        super(Wireframe, self).__init__()

    def __getitem__(self, index):
        with open(self.dir +self.train_data[index][:-3] + "pkl", "rb") as train_file:
            train_pkl = pickle.load(train_file)
            x_data = torch.from_numpy(train_pkl["fisheyeImg"].astype(np.float32))
            y_data = torch.Tensor([train_pkl["focalLength"]])
        return x_data, y_data

    def __len__(self):
        return self.train_data.shape[0]


if __name__ == "__main__":
    datasets = Wireframe("wireframe/v1.1/test1.txt")
    x, y = datasets[0]
    print(x, y)
    print(len(datasets))