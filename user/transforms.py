from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2
img_path = "data/train/ants/0013035.jpg"
img = Image.open(img_path)
cv_img = cv2.imread(img_path)


tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer = SummaryWriter("logs")
writer.add_image("Tensor_img", tensor_img)
print(tensor_img)


