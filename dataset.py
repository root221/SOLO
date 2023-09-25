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
        img = img /255.0
        img = torch.tensor(img)
        img = self.transform(img)

        mask = mask.astype(np.float32)
        mask = torch.tensor(mask)
        mask = transforms.Resize((800, 1066), interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask = transforms.Pad((11, 0))(mask)    
       
        
        scale = img.shape[-1] / 400 
        bbox *= scale
        bbox[:, [0, 2]] += 11  
        bbox = torch.tensor(bbox)

        # check flag
        assert img.shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.shape[0]
        return img, mask, bbox

def collate_fn(batch):
    images, labels, masks, bounding_boxes = list(zip(*batch))
    return torch.stack(images), labels, masks, bounding_boxes 
    
## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 2
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    #train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    #train_loader = train_build_loader.loader()
    #test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    #test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):
        import pdb
        pdb.set_trace()
        img, label, mask, bbox = [data[i] for i in range(len(data))]
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]


        # plot the origin img
        for i in range(batch_size):
            ## TODO: plot images with annotations
            plt.savefig("./testfig/visualtrainset"+str(iter)+".png")
            plt.show()

        if iter == 10:
            break

