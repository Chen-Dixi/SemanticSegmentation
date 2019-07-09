from data.utils import decode_segmap
import numpy as np
import torchvision.utils as vutils
import os
from PIL import Image
import torch
class Visualizer(object):
    def __init__(self,name):
        self.name=name

    def visualize_predict(self,root, pred,epoch):
        if not os.path.exists(root):
            os.mkdir(root)
        if pred.size(1) == 1:
            pred = torch.sigmoid(pred)
            pred = pred > 0.5
        else:
            pred = torch.argmax(pred, dim=1)
        for i in range(pred.size()[0]):
            mask = pred[i].detach().cpu().numpy()
            mask = np.array(mask).astype(np.uint8)
            segmap = decode_segmap(mask, dataset='cityscapes')
            
            segmap = np.array(segmap * 255).astype(np.uint8)
            im = Image.fromarray(segmap)
            im.save( os.path.join(root,"predict"+ str(i) +"epoch"+str(epoch)+".png") )
            


