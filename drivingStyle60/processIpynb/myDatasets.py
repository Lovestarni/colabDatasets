from torch.utils.data import DataLoader, Dataset
import torch
import h5py

class drivingStyleDataset(Dataset):
    def __init__(self, filePath, train=True, trainNum = 50):
        file = h5py.File(filePath, 'r')
        self.trainData = file['data'][:trainNum]
        self.trainLabel = file['label'][:trainNum]
        self.testData = file['data'][trainNum+1:]
        self.testLabel = file['label'][trainNum+1:]
        self.train = train
        #training set or testing set
    
    def __getitem__(self, idx):
        if self.train:
            data, label = self.trainData[idx], self.trainLabel[idx]
        else:
            data, label = self.testData[idx], self.testLabel[idx]
        return data, label
        
    def __len__(self):
        if self.train:
            return len(self.trainLabel)
        else:
            return len(self.testLabel)