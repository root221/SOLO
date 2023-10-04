import pytorch_lightning as pl
import torchvision
from solo_datamodule import SoloDataModule
from torch import optim
import torch
from torch import nn
import torch.nn.functional as F
from solo_branches import CategoryBranch, MaskBranch
from torchvision import transforms
import numpy as np
class SOLO(pl.LightningModule):
    _default_cfg = {
        'num_classes': 4,
        'in_channels': 256,
        'seg_feat_channels': 256,
        'stacked_convs': 7,
        'strides': [8, 8, 16, 32, 32],
        'scale_ranges': [(1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)],
        'epsilon': 0.2,
        'num_grids': [40, 36, 24, 16, 12],
        'mask_loss_cfg': dict(weight=3),
        'cate_loss_cfg': dict(gamma=2, alpha=0.25, weight=1),
        'postprocess_cfg': dict(cate_thresh=0.2, mask_thresh=0.5, pre_NMS_num=50, keep_instance=5, IoU_thresh=0.5)
    }

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in {**self._default_cfg, **kwargs}.items():
            setattr(self, k, v)

        pretrained_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)
        self.backbone = pretrained_model.backbone
        self.num_levels = len(self.scale_ranges)
        self.scale_ranges = torch.tensor(self.scale_ranges)
        self.cat_branch = CategoryBranch()
        self.mask_branch = MaskBranch(self.num_grids)
    # Forward function should calculate across each level of the feature pyramid network.
    # Input:
    #     images: batch_size number of images
    # Output:
    #     if eval = False
    #         category_predictions: list, len(fpn_levels), each (batch_size, C-1, S, S)
    #         mask_predictions:     list, len(fpn_levels), each (batch_size, S^2, 2*feature_h, 2*feature_w)
    #     if eval==True
    #         category_predictions: list, len(fpn_levels), each (batch_size, S, S, C-1)
    #         / after point_NMS
    #         mask_predictions:     list, len(fpn_levels), each (batch_size, S^2, image_h/4, image_w/4)
    #         / after upsampling
    def new_FPN(self, fpn_feat_list):
        #strides [8,8,16,32,32]
        #fpn_feat_list[0] =  
        return fpn_feat_list
      
    def MultiApply(self, func, *args, **kwargs):
        pfunc = partial(func, **kwargs) if kwargs else func
        map_results = map(pfunc, *args)
                 
        return tuple(map(list, zip(*map_results)))
    
    def forward_single_level(self, fpn_feat, idx, eval=False, upsample_shape=None):
        # upsample_shape is used in eval mode
        ## TODO: finish forward function for single level in FPN.
        ## Notice, we distinguish the training and inference.
        cate_pred = fpn_feat
        ins_pred = fpn_feat
        num_grid = self.num_grids[idx]  # current level grid

        # in inference time, upsample the pred to (ori image size/4)
        if eval == True:
            ## TODO resize ins_pred

            cate_pred = self.points_nms(cate_pred).permute(0,2,3,1)

        # check flag
        if eval == False:
            resized_feature = F.interpolate(fpn_feat, size=(self.num_grids[idx], self.num_grids[idx]), mode='bilinear') 
            cate_pred = self.cat_branch(resized_feature, idx)
            x, y = self.generate_pixel_coordinates(fpn_feat.shape[-2:])
            ins_pred = self.mask_branch(fpn_feat, x, y, idx)
            ins_pred = F.interpolate(ins_pred, scale_factor=2, mode='bilinear') 
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2)
        else:
            pass
        return cate_pred, ins_pred
        
    def forward(self, images, eval=True):
        # you can modify this if you want to train the backbone
        feature_pyramid = [v.detach() for v in self.backbone(images).values()] # this has strides [4,8,16,32,64]
        fpn_feat_list  = self.new_FPN(feature_pyramid) 
        cate_pred_list, ins_pred_list = self.MultiApply(self.forward_single_level, feature_pyramid, [i for i in range(len(feature_pyramid))])
        assert cate_pred_list[1].shape[2] == self.num_grids[1]
        assert ins_pred_list[1].shape[1] == self.num_grids[1]**2
        return cate_pred_list, ins_pred_list
        
    # This function build the ground truth tensor for each batch in the training
    # Input:
    #     bounding_boxes:   list, len(batch_size), each (n_object, 4) (x1 y1 x2 y2 system)
    #     labels:           list, len(batch_size), each (n_object, )
    #     masks:            list, len(batch_size), each (n_object, 800, 1088)
    # Output:
    #     category_targets: list, len(batch_size), list, len(fpn), (S, S), values are {1, 2, 3}
    #     mask_targets:     list, len(batch_size), list, len(fpn), (S^2, 2*feature_h, 2*feature_w)
    #     active_masks:     list, len(batch_size), list, len(fpn), (S^2,)
    #     / boolean array with positive mask predictions
    
    def compute_centre_regions(self, boxes_img):
        heights = boxes_img[:, 3] - boxes_img[:, 1]
        widths = boxes_img[:, 2] - boxes_img[:, 0]
        center_y = (boxes_img[:, 3] + boxes_img[:, 1])/2
        center_x = (boxes_img[:, 2] + boxes_img[:, 0])/2
            
        centre_regions_x1 = center_x - self.epsilon * widths / 2
        centre_regions_y1 = center_y - self.epsilon * heights / 2    
        centre_regions_x2 = center_x + self.epsilon * widths / 2
        centre_regions_y2 = center_y + self.epsilon * heights / 2
       
        centre_regions = torch.column_stack([centre_regions_x1, centre_regions_y1, centre_regions_x2, centre_regions_y2])
        
        return centre_regions    
    
    
    def generate_targets(self, bounding_boxes, labels, masks):
        # Bounding box format: [x_min, y_min, x_max, y_max] 
        fpn_len = len(self.scale_ranges)
        batch_size = len(bounding_boxes)
        img_height, img_width = masks[0].shape[-2:]
        category_targets = []
        mask_targets = []
        active_masks = []
        for boxes_img, labels_img, masks_img in zip(bounding_boxes, labels, masks):
            heights = boxes_img[:, 3] - boxes_img[:, 1]
            widths = boxes_img[:, 2] - boxes_img[:, 0]
                
            centre_regions = self.compute_centre_regions(boxes_img)
            
            sqrt_areas = (heights*widths).sqrt()
            scale_ranges = self.scale_ranges.to(self.device)
            fpn_level_masks = (sqrt_areas.unsqueeze(dim=1) >= scale_ranges[:, 0]) & (sqrt_areas.unsqueeze(dim=1) <= scale_ranges[:, 1])
            category_targets_per_img = []
            active_masks_per_img = []
            mask_targets_per_img = []
            for j in range(self.num_levels):        
                active_mask = torch.zeros((self.num_grids[j],self.num_grids[j])).to(self.device)
                category_target = torch.zeros((self.num_grids[j],self.num_grids[j])).to(self.device)
                feature_h = masks_img.shape[1] // self.strides[j]
                feature_w = masks_img.shape[2]// self.strides[j]  
                mask_target = torch.zeros(self.num_grids[j], self.num_grids[j], 2*feature_h, 2*feature_w).to(self.device)
                
                
                filtered_centre_regions = centre_regions[fpn_level_masks[:, j]] 
            
                grid_width = img_width / self.num_grids[j]
                grid_height = img_height / self.num_grids[j]
                if len(filtered_centre_regions) == 0:
                    active_masks_per_img.append(active_mask.reshape(-1))
                    category_targets_per_img.append(category_target)
                    mask_targets_per_img.append(mask_target.reshape(-1, 2*feature_h, 2*feature_w)) 
                    continue 
                x_grid_starts = (filtered_centre_regions[:, 0] / grid_width).floor().int()
                x_grid_ends = (filtered_centre_regions[:, 2] / grid_width).floor().int() 
                              
                y_grid_starts = (filtered_centre_regions[:, 1] / grid_height).floor().int()
                y_grid_ends = (filtered_centre_regions[:, 3] / grid_height).floor().int() 
              
                for k in range(len(x_grid_starts)):
                    x_grid_start = x_grid_starts[k]  
                    x_grid_end = x_grid_ends[k]
                    y_grid_start = y_grid_starts[k]
                    y_grid_end = y_grid_ends[k]  
                    active_mask[x_grid_start:x_grid_end+1, y_grid_start:y_grid_end+1] = 1
                    category_target[x_grid_start:x_grid_end+1, y_grid_start:y_grid_end+1] = labels_img[k]
                    resized_mask = F.interpolate(masks_img[k].view(1,1,img_height, -1), size=(2*feature_h, 2*feature_w), mode='nearest') 
                    mask_target[x_grid_start:x_grid_end+1, y_grid_start:y_grid_end+1] = resized_mask
                active_masks_per_img.append(active_mask.reshape(-1))
                category_targets_per_img.append(category_target)
                mask_targets_per_img.append(mask_target.reshape(-1, 2*feature_h, 2*feature_w))
                   
            mask_targets.append(mask_targets_per_img)
            category_targets.append(category_targets_per_img)  
            active_masks.append(active_masks_per_img)
        assert len(category_targets) == batch_size
        assert len(mask_targets) == batch_size
        assert len(active_masks) == batch_size
        assert len(category_targets[0]) == self.num_levels
        assert len(mask_targets[0]) == self.num_levels
        assert len(active_masks[0]) == self.num_levels 
        
        assert mask_targets[0][1].shape == (self.num_grids[1]**2, 200, 272)
        assert active_masks[0][1].shape == (self.num_grids[1]**2,)
        assert category_targets[0][1].shape == (self.num_grids[1], self.num_grids[1])
        return category_targets,mask_targets,active_masks 
    
    def generate_pixel_coordinates(self, size):
        h, w = size
        i_coord_channel = torch.linspace(-1, 1, h).unsqueeze(-1)
        j_coord_channel = torch.linspace(-1, 1, w).unsqueeze(0)
        i_coord_channel = i_coord_channel.repeat(1, w)
        j_coord_channel = j_coord_channel.repeat(h, 1)
        i_coord_channel = i_coord_channel.unsqueeze(0).unsqueeze(0)
        j_coord_channel = j_coord_channel.unsqueeze(0).unsqueeze(0)
        i_coord_channel = i_coord_channel.to(self.device)
        j_coord_channel = j_coord_channel.to(self.device)
        return i_coord_channel, j_coord_channel
        
    def loss(self,
             cate_pred_list,
             ins_pred_list,
             ins_gts_list,
             ins_ind_gts_list,
             cate_gts_list):
        
        num_level = len(self.num_grids)
        for i in range(num_level):
            self.focal_loss(cate_pred_list[i], cate_gts_list[i], i)   
     
        
    def focal_loss(self, cate_preds, cate_targets, level):
        ## TODO: compute focalloss
        import pdb
        pdb.set_trace()
        num_classes = 4
        alpha = self.cate_loss_cfg['alpha'] 
        gamma = self.cate_loss_cfg['gamma']
        cat_target_onehot = F.one_hot(torch.tensor(torch.stack(cate_targets),dtype=torch.int64),num_classes=num_classes)
        cat_target_onehot = cat_target_onehot[:,:,:,-3:].to(self.device)
        torch.where(cate_gts[0].to(torch.bool), alpha, 1 - alpha)
        pass
        
        
    def training_step(self, batch, batch_idx):
        imgs, labels, masks, bboxes = batch
        import pdb
        pdb.set_trace()
        
    
        category_targets, mask_targets, active_masks = self.generate_targets(bboxes, labels, masks)
        cate_pred_list, ins_pred_list = self(imgs)
        loss = self.loss(cate_pred_list, ins_pred_list, mask_targets, active_masks, category_targets)
        return loss

    def validation_step(self, batch, batch_idx):
        pass
        
    def on_validation_epoch_end(self):
        pass
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

        
if __name__ == '__main__':
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    datamodule = SoloDataModule(paths) 
    solo = SOLO()
    trainer = pl.Trainer(max_epochs=150, devices=1, precision=16)
    #trainer = pl.Trainer(max_epochs=150, devices=1)
    trainer.fit(solo, datamodule=datamodule)  
