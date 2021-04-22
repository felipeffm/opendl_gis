import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io, transform
import os 


#path_base = r"/home/fm/Documents/GitHub/em_desenvolvimento/GIS_c/1dataset"
#csv_file = os.path.join(path_base,'metadados_dataset.csv')

class CropedDataset(Dataset):
    """Cropped clipped raster dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with metadata.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.csv_path = os.path.join(self.root_dir,'metadados_dataset.csv')
        self.df_meta = pd.read_csv(self.csv_path)
        self.transform = transform
        self.n_classes = None
        self.updt_nclasses()

    def __len__(self):
        return len(self.df_meta)

    def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            #idx = 1
            img_path = str(self.df_meta.iloc[idx, 0])
            
            image = io.imread(img_path)
            
            image[image<0]=0
            
            classe = torch.tensor(int(self.df_meta.iloc[idx, 1]))
            sample = (image, classe)

            if self.transform:
                sample = (self.transform(sample[0]),classe)

            return sample
    def target(self):
        return self.df_meta['classe'].values
    
    def updt_nclasses(self):
        self.n_classes = len(set(self.target()))
    
#self = face_dataset        
#face_dataset = CropedDataset(csv_file=csv_file,root_dir=path_base)
