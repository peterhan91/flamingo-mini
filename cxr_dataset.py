import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode


class CXRDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, img_path, txt_path, column='findings', 
                 transform=None, target_transform=None):
        super().__init__()

        self.df = pd.read_csv(txt_path)
        self.col = column
        self.img_path = img_path
        self.df = self.df.filter(items=['dicom_id', column]).dropna()
        self.transform = transform
        self.target_transform = target_transform
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        dfs = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_path, dfs['dicom_id']+'.jpg'))
        img = img.convert('RGB')
        txt = dfs[self.col]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            txt = self.target_transform(txt)

        # print(img.shape, len(txt))
        # sample = {'img': img, 'txt': txt}
        return img, txt    
        


def load_data(cxr_filepath, txt_filepath, batch_size=4, column='report', pretrained=False, verbose=False): 
    if torch.cuda.is_available():  
        dev = "cuda:0" 
        cuda_available = True
        print('Using CUDA.')
    else:  
        dev = "cpu"  
        cuda_available = False
        print('Using cpu.')
    
    device = torch.device(dev)
    
    if cuda_available: 
        torch.cuda.set_device(device)

    if pretrained: 
        input_resolution = 224
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        ])
        print('Interpolation Mode: ', InterpolationMode.BICUBIC)
        print("Finished image transforms for pretrained model.")
    else: 
        input_resolution = 320
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        ])
        print("Finished image transforms for clip model.")
    
    torch_dset = CXRDataset(img_path=cxr_filepath,
                        txt_path=txt_filepath, column=column, transform=transform)
    
    if verbose: 
        for i in range(len(torch_dset)):
            sample = torch_dset[i]
            plt.imshow(sample['img'][0])
            plt.show()
            print(i, sample['img'].size(), sample['txt'])
            if i == 3:
                break
    
    loader_params = {'batch_size':batch_size, 'shuffle': True, 'num_workers': 0}
    data_loader = data.DataLoader(torch_dset, **loader_params)
    return data_loader, device