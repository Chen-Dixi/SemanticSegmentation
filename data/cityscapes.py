import torch.utils.data as data
import numpy as np

from PIL import Image
from torch.utils import data
from torchvision import transforms
import numpy as np
import sys, glob, os
sys.path.append("/home/apex/chendixi/Experiment/data/CityScapes")
import cityscapesscripts.helpers.labels as CityscapesLabels
import custom_transforms as tr

class CityscapesDataset(data.Dataset):

    num_classes = 19 #只训练需要predict的19个类

    #split in [train, val, test]
    def __init__(self,root, split="train",transform=None):
        self.root = root
        self.split = split
    
        self.files = []
        self.transform = transform
        

        if transform is None:
            self.default_transform()


        #leftImg8bit图片地址, image_dir已经指定了train，val，test
        self.image_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotation_dir = os.path.join(self.root, 'gtFine', self.split)

        searchLeftImg8bit   = os.path.join( self.image_dir , "*" , "*_leftImg8bit.png" )

        filesImage = glob.glob( searchLeftImg8bit )
        filesImage.sort()

        self.files = filesImage
        if not self.files:
            raise RuntimeError("No files for split=[%s] found in %s" % (split, self.image_dir))



    def __getitem__(self, index):
        
        path = self.files[index]
        labelImg_file = os.path.join(self.annotation_dir,
                                path.split(os.sep)[-2],
                                os.path.basename(path)[:-15] + 'gtFine_labelTrainIds.png')

        img = self.pil_loader(path)
        
        mask = np.array(Image.open(labelImg_file), dtype=np.uint8)
        mask = Image.fromarray(mask)
        sample = {'image': img, 'label': mask}
        if self.transform is not None:
            sample = self.transform(sample)
        
            

        return sample['image'], sample['label']

    def __len__(self):
        return len(self.files)

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def default_transform(self):
        self.transform = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.FixedResize(size=513),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
       
    
