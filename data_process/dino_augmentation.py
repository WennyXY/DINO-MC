
from torchvision import transforms
import utils.utils as utils
from PIL import ImageFilter

class DataAugColorMulticrop(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, size_crops, min_scale_crops, max_scale_crops):
        color_transform = [get_color_distortion(), utils.GaussianBlur()]
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        assert(len(size_crops) >= local_crops_number)
        trans = []
        if len(size_crops) > local_crops_number:
            size_crops = size_crops[len(size_crops)-local_crops_number:]
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops, max_scale_crops),
            )
            trans.append(transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                normalize
            ]))
        self.trans = trans

    def __call__(self, image):
        crops = []
        if type(image).__name__ =='list' and len(image) > 1:
            crops.append(self.global_transfo1(image[0]))
            crops.append(self.global_transfo2(image[1]))
            multi_crops = list(map(lambda trans: trans(image[2]), self.trans))
        else:
            crops.append(self.global_transfo1(image))
            crops.append(self.global_transfo2(image))
            # size_crops = self.size_crops
            
            multi_crops = list(map(lambda trans: trans(image), self.trans))
        crops.extend(multi_crops)
        return crops

class DataAugmentationTP(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, size_crops, min_scale_crops, max_scale_crops):
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.global_crop = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            normalize
        ])
        trans = []
        if len(size_crops) > local_crops_number:
            size_crops = size_crops[len(size_crops)-local_crops_number:]

        assert(len(size_crops) == local_crops_number)
        for i in range(len(size_crops)):
            trans.append(transforms.Compose([
                transforms.RandomResizedCrop(
                    size_crops[i],
                    scale=(min_scale_crops, max_scale_crops),
                ),
                normalize
            ]))
        self.trans = trans

    def __call__(self, image):
        crops = []
        crops.append(self.global_crop(image[1]))
        crops.append(self.global_crop(image[2]))
        crops.append(self.global_crop(image[3]))
        multi_crops = list(map(lambda trans: trans(image[0]), self.trans))
        crops.extend(multi_crops)
        return crops


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
