import torch
import torch.utils.data

class PCNNDataSet(torch.utils.data.Dataset):
    def __init__(self, data, mask, label):
        self.data = torch.FloatTensor(data)
        self.mask = torch.FloatTensor(mask)
        self.label = torch.LongTensor(label)

    def __getitem__(self, index):
        return self.data[index], self.mask[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]