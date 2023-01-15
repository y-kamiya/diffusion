import torch


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, dim=4, num_max=8):
        self.dim = dim
        self.num_max = num_max

        data = []
        for i in range(num_max - dim + 1):
            data.append(list(range(i, i + dim)))
            data.append(list(range(i + dim - 1, i - 1, -1)))

        self.data = self.normalize(torch.Tensor(data)).repeat(1000, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def normalize(self, x):
        return 2 * x / (self.num_max - 1) - 1

    def denormalize(self, x):
        return (x + 1) * (self.num_max - 1) / 2
