import torch
import torch.nn.functional as F
import torch.nn as nn
smooth_l1_sigma = 1.0
smooth_l1_loss = nn.SmoothL1Loss(reduction='none')    # reduce=False


def ohem_loss(batch_size, cls_pred, cls_target, loc_pred, loc_target):   
    """    Arguments:
     batch_size (int): number of sampled rois for bbox head training      
     loc_pred (FloatTensor): [R, 4], location of positive rois      
     loc_target (FloatTensor): [R, 4], location of positive rois   
     pos_mask (FloatTensor): [R], binary mask for sampled positive rois   
     cls_pred (FloatTensor): [R, C]     
     cls_target (LongTensor): [R]  
     Returns:    
           cls_loss, loc_loss (FloatTensor)
    """

    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)
    ohem_loc_loss = smooth_l1_loss(loc_pred, loc_target).sum(dim=1)
    # 这里先暂存下正常的分类loss和回归loss
    print(ohem_cls_loss.shape, ohem_loc_loss.shape)
    loss = ohem_cls_loss + ohem_loc_loss
    # 然后对分类和回归loss求和
    
    sorted_ohem_loss, idx = torch.sort(loss, descending=True)   
    # 再对loss进行降序排列
    
    keep_num = min(sorted_ohem_loss.size()[0], batch_size)    
    # 得到需要保留的loss数量
    
    if keep_num < sorted_ohem_loss.size()[0]:    
        # 这句的作用是如果保留数目小于现有loss总数，则进行筛选保留，否则全部保留
    
        keep_idx_cuda = idx[:keep_num]        # 保留到需要keep的数目
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]      
        ohem_loc_loss = ohem_loc_loss[keep_idx_cuda]        # 分类和回归保留相同的数目
        
    cls_loss = ohem_cls_loss.sum() / keep_num   
    loc_loss = ohem_loc_loss.sum() / keep_num    # 然后分别对分类和回归loss求均值
    return cls_loss, loc_loss


if __name__ == '__main__':
    batch_size = 4
    C = 6
    loc_pred = torch.randn(8, 4)
    loc_target = torch.randn(8, 4)
    cls_pred = torch.randn(8, C)
    cls_target = torch.Tensor([1, 1, 2, 3, 5, 3, 2, 1]).type(torch.long)
    cls_loss, loc_loss = ohem_loss(batch_size, cls_pred, cls_target, loc_pred, loc_target)
    print(cls_loss, '--', loc_loss)