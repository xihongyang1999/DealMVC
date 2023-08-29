import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import h5py

data_info = dict(
    BBCSport = {1: 'BBCSport', 'N': 544, 'K': 5, 'V': 2, 'n_input': [3183,3203], 'n_hid': [512,512], 'n_output': 64},
)

class GetData(Dataset):
    def __init__(self, name):
        data_path = './data/{}.mat'.format(name[1])
        np.random.seed(1)
        index = [i for i in range(name['N'])]
        np.random.shuffle(index)

        data = h5py.File(data_path)
        Final_data = []
        for i in range(name['V']):
            diff_view = data[data['X'][0, i]]
            diff_view = np.array(diff_view, dtype=np.float32).T
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y']).T
        LABELS = label[index]
        self.name = name
        self.data = Final_data
        self.y = LABELS

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if self.name['V'] == 2:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx])], torch.from_numpy(
                self.y[idx]), torch.from_numpy(np.array(idx)).long()
        elif self.name['V'] == 3:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]),
                    torch.from_numpy(self.data[2][idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(
                np.array(idx)).long()
        elif self.name['V'] == 4:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]),
                    torch.from_numpy(self.data[2][idx]), torch.from_numpy(self.data[3][idx])], \
                   torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
        elif self.name['V'] == 5:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]),
                    torch.from_numpy(self.data[2][idx]), torch.from_numpy(self.data[3][idx]),
                    torch.from_numpy(self.data[4][idx])], \
                   torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
        elif self.name['V'] == 6:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]),
                    torch.from_numpy(self.data[2][idx]), torch.from_numpy(self.data[3][idx]),
                    torch.from_numpy(self.data[4][idx]), torch.from_numpy(self.data[5][idx])], \
                   torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
        elif self.name['V'] == 7:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]),
                    torch.from_numpy(self.data[2][idx]), torch.from_numpy(self.data[3][idx]),
                    torch.from_numpy(self.data[4][idx]), torch.from_numpy(self.data[5][idx]),
                    torch.from_numpy(self.data[6][idx])], \
                   torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
        else:
            raise NotImplementedError


def load_data(data_name):
    dataset_para = data_info[data_name]
    dataset = GetData(dataset_para)
    dims = dataset_para['n_input']
    view = dataset_para['V']
    data_size = dataset_para['N']
    class_num = dataset_para['K']
    return dataset, dims, view, data_size, class_num