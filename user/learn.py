# 常见的Transforms
# 输入 *PIL
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter('logs1')
image_path = 'data/train/ants/6240329_72c01e663e.jpg'
img = Image.open(image_path)
print(img)

#ToTensor将pil对象转化为tensor对象
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

#Tormallize图片其实是一个三维数组，每个像素都是一个rgb向量，该函数通过设置mean均值和std来实现正则化
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5 ,0.5])
img_norm = trans_norm.forward(img_tensor)
print(img_tensor[0][0][0])
writer.add_image("Normlize", img_norm)

#Resize改变图片的大小
print(img.size)
trans_resize = transforms.Resize((256, 256))
img_resize = trans_resize.forward(img)
img_resizetotenser = trans_totensor(img_resize)
writer.add_image("Resize", img_resizetotenser, 0)
print(img_resize)

#Compose
trans_resize1 = transforms.Resize(256)
trans_compose = transforms.Compose([trans_resize1, trans_totensor])
img_resizetotensor = trans_compose(img)
writer.add_image("Resize1", img_resizetotensor, 1)
writer.close()