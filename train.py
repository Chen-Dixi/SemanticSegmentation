from blseg import model, loss, metric
from config import *
from data.cityscapes import *
from torch import optim
import torch
import torch.nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dixitool.pytorch.module import functional as F
from torch.optim.lr_scheduler import StepLR
from utils.visualize import Visualizer

output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
visualizer = Visualizer("asd")

def train(net, criterion, train_dataloader, optimizer, lr_scheduler, epoch):
    print("training")
    net.train()
    train_loss = 0.0
    for i, sample in enumerate(tqdm(train_dataloader)):
        images, targets = sample
        images = images.to(output_device)
        targets = targets.to(output_device)
        optimizer.zero_grad()

        # 暂时不考虑调整learning rate
        outputs = net(images)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    lr_scheduler.step(epoch)
    print("epoch {epoch} loss:{loss}".format(epoch=epoch,loss=train_loss))

best_miou = 0.0

#每evaluate一段时间就看一下结果
#画图
def validation(net, criterion, val_dataloader, optimizer, mIoU_metric, lr_scheduler, epoch):
    print("evaluating")
    net.eval()
    test_loss = 0.0
    mIoU_metric.reset()
    global best_miou
    
    for i, sample in tqdm(enumerate(val_dataloader)):
        images, targets = sample
        images = images.to(output_device)
        targets = targets.to(output_device)
        with torch.no_grad():
            outputs = net(images)
        loss = criterion(outputs, targets.long())
        test_loss += loss.item()
        #每个batch更新一次metric
        mIoU_metric.update(outputs, targets)

        if i == args.val.visualize_batch_index:
            # 模型预测结果导出图片，只导出第一个batch的第一张图片
            visualizer.visualize_predict('result', outputs, epoch, mode='first')

    miou = mIoU_metric.get()
    print("epoch {epoch} miou:{miou}".format(epoch=epoch,miou=miou))
    
    #保存最优模型
    if miou > best_miou:
        best_miou = miou
        F.save_model('checkpoints', 'mIoU[%.2f]' % best_miou, net)

    #保存存档点
    #保存的内容， lr_scheduler, optimizer, net, epoch, best_mIoU
    F.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'best_pred': best_miou,
            }, False,'checkpoints' )

def main(args):
    
    net = model.DeepLabV3Plus(backbone='mobilenet_v2', num_classes=args.data.num_classes)
    net = net.to_device_(args.train.num_gpus)
    ohem_ratio = 0.7
    criterion = loss.CrossEntropyLossWithOHEM(ohem_ratio, ignore_index=255)
    mIoU = metric.MeanIoU(args.data.num_classes)

    
    dataset = CityscapesDataset(args.data.data_dir,split='train')
    dataloader = DataLoader(dataset, batch_size=args.data.batch_size, shuffle=True)

    val_dataset = CityscapesDataset(args.data.data_dir,split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.data.batch_size)

    optimizer = optim.SGD([{"params":net.parameters()}],
                          lr=args.train.lr,
                          weight_decay=args.train.weight_decay)
    lr_scheduler = StepLR(optimizer,args.train.scheduler_step_size)

    epochs = args.train.epochs

    # it = iter(val_dataloader)
    # images,targets = it.next()
    # images = images.to(output_device)
    # targets = targets.to(output_device)
    # with torch.no_grad():
    #     outputs = net(images)
    # print(outputs.size())#torch.Size([8, 19, 512, 512])
    for epoch in range(epochs):
        print("Epoch:{}/{}".format(epoch, epochs))
        train(net, criterion, dataloader, optimizer, lr_scheduler, epoch)
        if epoch % args.train.eval_interval == (args.train.eval_interval - 1):
            validation(net, criterion, val_dataloader, optimizer, mIoU, lr_scheduler, epoch)

if __name__ == '__main__':
    main(args)

