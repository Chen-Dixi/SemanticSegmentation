from data.utils import decode_segmap
import numpy as np
import torchvision.utils as vutils
import os
from PIL import Image
import torch
class Visualizer(object):
    def __init__(self,name):
        self.name=name

    def visualize_predict(self,root, pred,epoch ,mode='first'):
        assert mode in ['first', 'all' , 'last']

        if not os.path.exists(root):
            os.mkdir(root)
        if pred.size(1) == 1:
            pred = torch.sigmoid(pred)
            pred = pred > 0.5
        else:
            pred = torch.argmax(pred, dim=1)

        if mode == 'all':    
            for i in range(pred.size()[0]):
                mask = pred[i].detach().cpu().numpy()
                mask = np.array(mask).astype(np.uint8)
                self.save_predict(mask, os.path.join(root,"predict"+ str(i) +"epoch"+str(epoch)+".png"))
                # segmap = decode_segmap(mask, dataset='cityscapes')
                
                # segmap = np.array(segmap * 255).astype(np.uint8)
                # im = Image.fromarray(segmap)
                # im.save( os.path.join(root,"predict"+ str(i) +"epoch"+str(epoch)+".png") )
        elif mode == 'first':
            mask = pred[0].detach().cpu().numpy()
            mask = np.array(mask).astype(np.uint8)
            self.save_predict(mask, os.path.join(root,"predict_epoch"+str(epoch)+".png") )
        else:
            mask = pred[-1].detach().cpu().numpy()
            mask = np.array(mask).astype(np.uint8)
            self.save_predict(mask, os.path.join(root,"predict_epoch"+str(epoch)+".png"))

    def save_predict(self, mask, path):
        
        segmap = decode_segmap(mask, dataset='cityscapes')
        
        segmap = np.array(segmap * 255).astype(np.uint8)
        im = Image.fromarray(segmap)
        im.save( path)

            


