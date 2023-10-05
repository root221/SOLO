## Author: Lishuo Pan 2020/4/18

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # TODO: load dataset, make mask list
        imgs_path, masks_path, labels_path, bboxes_path = path
        
        self.imgs = h5py.File(imgs_path, 'r')['data']

        self.masks = h5py.File(masks_path, 'r')['data']

        self.transform = transforms.Compose([
            transforms.Resize((800, 1066)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Pad((11, 0)) 
        ])
        self.labels = np.load(labels_path, allow_pickle=True)
        self.bboxes = np.load(bboxes_path, allow_pickle=True)
        self.cumulative_labels = np.cumsum([0] + [len(label) for label in self.labels])
        
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
    def __getitem__(self, index):
        # TODO: __getitem__
        raw_img = self.imgs[index]
        mask_init_index = self.cumulative_labels[index]
        mask_end_index = self.cumulative_labels[index+1] 
        raw_mask = self.masks[mask_init_index:mask_end_index]  
        raw_bbox = self.bboxes[index]
        label = self.labels[index]
        # Preprocess the raw data
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(raw_img, raw_mask, raw_bbox)
        label = torch.tensor(label)
        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox
        
        
    def __len__(self):
        #return len(self.imgs_data)
        return len(self.labels)
    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        # TODO: image preprocess
        original_width = img.shape[-1]
        img = img /255.0
        img = torch.tensor(img, dtype=torch.float32)
        img = self.transform(img)

        mask = mask.astype(np.float32)
        mask = torch.tensor(mask)
        mask = transforms.Resize((800, 1066), interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask = transforms.Pad((11, 0))(mask)    
       
        scale = 1066 / original_width 
        bbox = torch.tensor(bbox)
        bbox *= scale
        bbox[:, [0, 2]] += 11  
    
        # check flag
        assert img.shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.shape[0]
        return img, mask, bbox

def collate_fn(batch):
    images, labels, masks, bounding_boxes = list(zip(*batch))
    return torch.stack(images), labels, masks, bounding_boxes 
    

