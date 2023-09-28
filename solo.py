import pytorch_lightning as pl
import torchvision
from solo_datamodule import SoloDataModule
from torch import optim
import torch
from torch import nn
import torch.nn.functional as F
from solo_branches import CategoryBranch, MaskBranch 
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
        self.category_branch = CategoryBranch()
        self.mask_branch = MaskBranch()
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
            cate_pred = self.category_branch(resized_feature)
            ins_pred = self.mask_predictions(fpn_feat)
            assert cate_pred.shape[1:] == (3, num_grid, num_grid)
            assert ins_pred.shape[1:] == (num_grid**2, fpn_feat.shape[2]*2, fpn_feat.shape[3]*2)
        else:
            pass
        return cate_pred, ins_pred
        
    def forward(self, images, eval=True):
        # you can modify this if you want to train the backbone
        feature_pyramid = [v.detach() for v in self.backbone(images).values()] # this has strides [4,8,16,32,64]
        fpn_feat_list  = self.new_FPN(feature_pyramid) 
        self.forward_single_level(fpn_feat_list[2], 2)
        import pdb
        pdb.set_trace() 
        assert cate_pred_list[1].shape[2] == self.num_grids[1]
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
        center_x = (boxes_img[:, 2] + boxes_img[:, 1])/2
            
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
        for boxes_img, labels_img, masks_img in zip(bounding_boxes, labels, masks):
            heights = boxes_img[:, 3] - boxes_img[:, 1]
            widths = boxes_img[:, 2] - boxes_img[:, 0]
                
            centre_regions = self.compute_centre_regions(boxes_img)
            
            sqrt_areas = (heights*widths).sqrt()
            scale_ranges = self.scale_ranges.to(self.device)
            fpn_level_masks = (sqrt_areas.unsqueeze(dim=1) >= scale_ranges[:, 0]) & (sqrt_areas.unsqueeze(dim=1) <= scale_ranges[:, 1])
            for j in range(self.num_levels):        
                active_mask = torch.zeros((self.num_grids[j]*self.num_grids[j])).to(self.device)
                filtered_centre_regions = centre_regions[fpn_level_masks[:, j]] 
                
                grid_width = img_width / self.num_grids[j]
                x_grid_start = (filtered_centre_regions[:, 0] / grid_width).ceil().int()
                x_grid_end = (filtered_centre_regions[:, 2] / grid_width).floor().int() 
                              
                
#active_mask = 
   
    
    def training_step(self, batch, batch_idx):
        imgs, labels, masks, bboxes = batch
        self.generate_targets(bboxes, labels, masks)
        self(imgs)
        loss = 0
        return loss

    def validation_step(self, batch, batch_idx):
        pass
        
    def on_validation_epoch_end(self):
        pass
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss(self):
        pass     
        
if __name__ == '__main__':
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    datamodule = SoloDataModule(paths) 
    solo = SOLO()
    trainer = pl.Trainer(max_epochs=150, devices=1)
    trainer.fit(solo, datamodule=datamodule)  
