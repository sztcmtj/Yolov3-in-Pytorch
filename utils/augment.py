from torchvision import transforms as trans
from utils.vis_utils import *
from utils.box_utils import *
from utils.dataset_tools import *
from utils.utils import *
from imgaug import augmenters as iaa

class Aug_Yolo(object):
    def __init__(self,rotate=0,shear=0,PerspectiveTransform=0):
        self.rotate = rotate
        self.shear = shear
        self.PerspectiveTransform = PerspectiveTransform
    
    def head_tsfm(self,conf):
        seed = random.randint(0,1e8)
        iaa_ins = iaa.Sequential([
            iaa.Affine(rotate=[0,self.rotate],random_state = seed, deterministic=True),
            iaa.Fliplr(0.5,random_state = seed, deterministic=True)
        ])
        transform = trans.Compose([
            trans.Lambda(lambda x: imgaug_on_PIL(iaa_ins,x)),
            trans.Resize([conf.input_size,conf.input_size]),
            trans.ToTensor(),
            trans.Normalize(conf.mean, conf.std)
        ])
        return transform
    
    def aff_tsfm(self,conf):
        seed = random.randint(0,int(1e6))
        
        iaa_img = iaa.Sequential([
            iaa.OneOf([
                iaa.OneOf([
                    iaa.Affine(rotate=[0,self.rotate], random_state = seed, deterministic=True, mode='edge'),
                    iaa.Affine(shear=[0,self.shear], random_state = seed, deterministic=True, mode='edge')
                ], random_state = seed, deterministic=True),
                iaa.PerspectiveTransform(self.PerspectiveTransform, random_state = seed, deterministic=True)
            ], random_state = seed, deterministic=True),            
            iaa.Fliplr(0.5,random_state = seed, deterministic=True)
        ], random_state = seed, deterministic=True, random_order = True)
        
        iaa_mask = iaa.Sequential([
            iaa.OneOf([
                iaa.OneOf([
                    iaa.Affine(rotate=[0,self.rotate], random_state = seed, deterministic=True),
                    iaa.Affine(shear=[0,self.shear], random_state = seed, deterministic=True)
                ], random_state = seed, deterministic=True),
                iaa.PerspectiveTransform(self.PerspectiveTransform, random_state = seed, deterministic=True)
            ], random_state = seed, deterministic=True),            
            iaa.Fliplr(0.5,random_state = seed, deterministic=True)
        ], random_state = seed, deterministic=True, random_order = True)
        
        tsfm_img = trans.Compose([
            trans.Lambda(lambda x: imgaug_on_PIL(iaa_img,x)),
            trans.Resize([conf.input_size,conf.input_size])
        ])
        
        tsfm_mask = trans.Compose([
            trans.Lambda(lambda x: imgaug_on_PIL(iaa_mask,x)),
            trans.Resize([conf.input_size,conf.input_size])
        ])
        
        return tsfm_img,tsfm_mask
    
    def noise_tsfm(self,conf):
        seq = iaa.Sequential([
            iaa.OneOf(
                children=[
                    iaa.GaussianBlur((0., 1.2)),
                    iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
                    iaa.Emboss(alpha=(0, 0.5), strength=(0, 1.0)),
                    iaa.ElasticTransformation(0.8),
                    iaa.OneOf([
                        iaa.CoarseSaltAndPepper(p=(0., 0.15), size_percent=0.3),
                        iaa.Dropout(p=0.15)
                    ])
                ])
            ])
        transform = trans.Compose([
            trans.Lambda(lambda x: imgaug_on_PIL(seq,x)),
            trans.RandomApply([trans.ColorJitter(0.1,0.15,0.15)]),
            trans.RandomApply([trans.ColorJitter(hue=0.1)]),
            trans.ToTensor(),
            trans.Normalize(conf.mean, conf.std)
        ])
        return transform    
    
def imgaug_on_PIL(iaa_ins,img):
    return Image.fromarray(iaa_ins.augment_image(np.asarray(img)))