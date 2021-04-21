from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# Load the image
img = Image.open('horse.jpg')

# show the type and size.
print(type(img), img.size)
# for fun show the image.
# img.show()
# for fun, we'll first go to a Numpy array
img_a = np.array(img)
# Let's look at our shape
# It should be H x W x C, height, width, channels.
# In this case, we are using RGB, so we have 3 channels.
print(img_a.shape)
# PyTorch modules dealing with image data require tensors to be laid out as
# C x H x W, channels, height, and width
# we can rearrange the layout, but first lets go to a Tensor
img_t = torch.from_numpy(img_a)
img_t = img_t.permute(2, 0, 1)
# some times we could have another channel like an an alpha channel, for transparency. for now, we only care
# about RGB
img_t = img_t[:3]

# now look at the shape again
print(img_t.shape)
# we are using bytes (0 to 255), but typically we'll want to use floats for torchvision
img_t = img_t.float()
# tensor is now filled with floats
# what does the image look like now?
out_img = transforms.ToPILImage()(img_t)
out_img.show()

# does the image look ok?  It shouldn't.
# Pillow handles images with floats and bytes slight differently
# even though they have the same value.
# This can be fixed by scaling out data to be (in this case) between 0 and 1.

img_t /= 255.0
out_img = transforms.ToPILImage()(img_t)
out_img.show()
# all better now

# we can do some other fun stuff as well, like transform the image
# in this cass, resize and center crop
process = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])
img_t = process(img_t)
# look at our shape
print(img_t.shape)
out_img = transforms.ToPILImage()(img_t)
out_img.show()

# transforms is really great and not only does it work with Tensors
# it also works with Pillow images...
# This means we can use Pillow images directly

process = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

pillow_to_tensor = process(img)
# lets look at what we got back
print(type(pillow_to_tensor),pillow_to_tensor.shape,pillow_to_tensor.dtype)
# and the image.

out_img = transforms.ToPILImage()(pillow_to_tensor)
out_img.show()
