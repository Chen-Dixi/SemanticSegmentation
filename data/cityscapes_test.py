import unittest
from cityscapes import CityscapesDataset
from torch.utils.data import DataLoader
import tqdm
from PIL import Image
from utils import decode_segmap
import numpy as np
import torchvision.utils as vutils
import torch



class CityscapesTestCase(unittest.TestCase):
    # def test_load(self):
        
    #     TEXT_DATA_DIR='20_newsgroups'
    #     news_ds = Newsgroup(TEXT_DATA_DIR,remove=('headers','footers','quotes'))
    #     print(news_ds[0])

    def set_dataset(self):
        DATA_DIR='/home/apex/chendixi/Experiment/data/CityScapes'
        dataset = CityscapesDataset(DATA_DIR,split='train')
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        it = iter(dataloader)
        image,mask = it.next()
        # file = dataset.files[0]

    def test_img8bit_mask(self):
        DATA_DIR='/home/apex/chendixi/Experiment/data/CityScapes'
        dataset = CityscapesDataset(DATA_DIR,split='train')
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        it = iter(dataloader)
        images,masks = it.next()
        #masks 并没有被除以255
        for i in range(images.size()[0]):
            vutils.save_image(images[i],"image"+ str(i) +".png")
            mask = masks[i].numpy()

            tmp = np.array(mask).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            segmap = torch.from_numpy(segmap).float().permute(2, 0, 1)
            vutils.save_image(segmap,"label"+ str(i) +".png")

    
    def save_mask(self):
        DATA_DIR='/home/apex/chendixi/Experiment/data/CityScapes'
        dataset = CityscapesDataset(DATA_DIR,split='train')
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        it = iter(dataloader)
        images,masks = it.next()
        #masks 并没有被除以255
        
        mask = masks[1].numpy()

        tmp = np.array(mask).astype(np.uint8)
        segmap = decode_segmap(tmp, dataset='cityscapes')
        segmap = np.array(segmap * 255).astype(np.uint8)
        im = Image.fromarray(segmap)
        im.save("label.png")

if __name__ == '__main__':
    unittest.main()