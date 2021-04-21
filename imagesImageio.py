import imageio
import torch
from torchvision import transforms
from PIL import Image
# load the image

img_arr = imageio.imread('horse.jpg')

print(type(img_arr))
# “img is a NumPy array-like object with three dimensions,
# two spatial dimensions, width and height; and a third dimension corresponding to RGB channels.
# we can get our shape
print(img_arr.shape)
# change to a tensor...
img = torch.from_numpy(img_arr)
img_t = img.permute(2, 0, 1)

# should all work
grayscale_image = 0.4 * img_t[0] + 0.4 * img_t[1] + 0.2 * img_t[2]
# grayscale_image = (r_image + g_image + b_image).div(3.0)
img_t[0] = grayscale_image
img_t[1] = grayscale_image
img_t[2] = grayscale_image

out_img = transforms.ToPILImage()(img_t)
out_img.show()

