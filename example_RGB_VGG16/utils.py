import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io, transform
import os 
import rasterio
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import torchvision
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
        self._updt_nclasses()
        
    def __len__(self):
        return len(self.df_meta)
    
    def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            #idx = 1
            img_path = str(self.df_meta.iloc[idx, 0])

            if img_path.endswith('tif'):
                #img_path=r"/home/fm/Documents/GitHub/em_desenvolvimento/GIS_c/dataset/0.0/croppid_0-class_0.0-shp_base_treino_CAN_ROv4-raster_6955-74604.tif"
                image = rasterio.open(img_path)
                image = np.nan_to_num(image.read())
                image = torch.tensor(image)
                

            elif img_path.endswith('jpeg'):
                #img_path=r"/home/fm/Documents/GitHub/em_desenvolvimento/GIS_c/dataset_jpeg/0.0/croppid_0-class_0.0-shp_base_treino_CAN_ROv4-raster_6955-74604.jpeg"
                
                image = io.imread(img_path)
                image = image.astype(float)
                image = ToTensor()(image)
                #np.moveaxis(image,-1,0).shape
                #image = torch.tensor(image)
                #image = read_image(img_path,mode=torchvision.io.image.ImageReadMode.RGB).double()
                

            
                
            
            
            classe = torch.tensor(int(self.df_meta.iloc[idx, 1]))
            sample = (image, classe)

            if self.transform:
                sample = (self.transform(sample[0]),classe)

            return sample

    def target(self):
        return self.df_meta['classe']
    
    def _updt_nclasses(self):
        self.n_classes = len(set(self.target()))
"""
DEBUG
path_base = r"/home/fm/Documents/GitHub/em_desenvolvimento/GIS_c/dataset_jpeg"
csv_file = os.path.join(path_base,'metadados_dataset.csv')
df_meta = pd.read_csv(csv_file)
filenames = pd.read_csv(csv_file)['filename']
hist = []
io.imread(filename_)
for filename_ in filenames:
    filename_ = 
    tensor_img = read_image(filename_)
    tensor_img.astype(float)
    tensor_np_img = tensor_img.numpy()
    
    np_img.min()
    tensor_np_img.mean()
    tensor_np_img.min()
    np_img.dtype
    tensor_np_img.dtype
    2**8
    image = rasterio.open(filename_)
    image = np.nan_to_num(image.read())
    image.shape
    #hist.append(image.shape)
    #image_ = np.moveaxis(image, source = -1, destination= 0)
    image = ToTensor()(image).shape
    
    image.shape
    


xx = CropedDataset(path_base)

for i in range(len(filenames)):
    
    hist.append(xx.__getitem__(0)[0].shape)

xx.__getitem__(4)[0].shape
xx.__getitem__(3)[0].shape
xx.__getitem__(2)[0].shape
img = xx.__getitem__(1)[0]
shape = img.shape
.shape


self = xx

idx = 0
"""