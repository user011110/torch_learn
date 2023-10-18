import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset",False,transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
#测试集中第一张图片及target
img,target=test_data[0]
print(img.shape)
print(target)
#dataloader将dataset中数据进行打包
writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs ,targets = data
    writer.add_images("test_data", imgs, step)
    step=step+1
writer.close()