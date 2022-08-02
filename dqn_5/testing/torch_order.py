
import numpy as np
import torch

# currently H, W, C
_image = [[[1]]]

# convert to C, H, W
_image = np.array(_image)
image = torch.from_numpy(_image)
image = image[np.newaxis, :]

print(image)
print(image.shape)
