import unittest
from data import CityscapesDataset
from torch.utils.data import DataLoader
import tqdm
from PIL import Image
from data.utils import decode_segmap
import numpy as np
import torchvision.utils as vutils
import torch
from utils.visualize import Visualizer
class VisualizeTestCase(unittest.TestCase):
    # def test_load(self):
        
    #     TEXT_DATA_DIR='20_newsgroups'
    #     news_ds = Newsgroup(TEXT_DATA_DIR,remove=('headers','footers','quotes'))
    #     print(news_ds[0])

    

    def test_output_visualize(self):
        vs = Visualizer("asd")
        DATA_DIR='/home/apex/chendixi/Experiment/data/CityScapes'
        dataset = CityscapesDataset(DATA_DIR,split='train')
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        it = iter(dataloader)
        images,masks = it.next()
        #masks 并没有被除以255
        print(images.size()) #torch.Size([2, 3, 513, 513])
        vs.visualize_predict('result', images,0)

if __name__ == '__main__':
    unittest.main()