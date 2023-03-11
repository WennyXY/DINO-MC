import os
from PIL import Image
import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils
import torchvision.transforms as transforms

def saveImg(img, img_path, Gray=True):
    # fname, fext = name.split('.')
    # imgPath = os.path.join(save_dir, "%s_%s.%s" % (fname, type, fext))
    # torchvision.utils.save_image(img, imgPath)
    # 改写：torchvision.utils.save_image
    grid = torchvision.utils.make_grid(img, nrow=8, padding=2, pad_value=0,
                                    normalize=False, range=None, scale_each=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    # im.show()
    if Gray:
        im.convert('L').save(img_path)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
    else:
        im.save(img_path)

def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis

def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5

def visualize(tensor_list, denorm=False):
    vis_list = []
    for ten in tensor_list:
        if ten.dim() == 4:
            ten = ten[0]
        if denorm:
            ten = de_norm(ten)
        img = make_numpy_grid(ten)
        vis_list.append(img)
    vis = np.concatenate(vis_list, axis=0)
    vis = np.clip(vis, a_min=0.0, a_max=1.0)
    return vis


        

def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:,None,None]).add_(mean[:,None,None])
        
    img_tensor = img_tensor.transpose(0,2).transpose(0,1)  # C x H x W  ---> H x W x C
    
    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy()*255
    
    if isinstance(img_tensor, torch.Tensor):
    	img_tensor = img_tensor.numpy()
    
    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))
        
    return img