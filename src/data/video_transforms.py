import torch
import torchvision.transforms.functional as TF
import random


class RandomRotateVideo(object):
    def __init__(self, degrees, expand=False, center=None, fill=0):
        if type(degrees) == int or type(degrees) == float:
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees_low = -degrees
            self.degrees_high = degrees
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be (min_angle, max_angle).")
            self.degrees_low = degrees[0]
            self.degrees_high = degrees[1]

        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, x):
        # this is kinda sketchy but I didn't want to write a method
        # that generalizes to PIL video
        assert torch.is_tensor(x)
        if x.dim() == 4:
            c, t, h, w = x.shape
            angle = random.uniform(self.degrees_low, self.degrees_high)
            for i in range(t):
                x[:, i, :, :] = TF.rotate(
                    x[:, i, :, :],
                    angle,
                    expand=self.expand,
                    center=self.center,
                    fill=self.fill,
                )
        elif x.dim() == 3:
            t, h, w = x.shape
            angle = random.uniform(self.degrees_low, self.degrees_high)
            for i in range(t):
                x[i, :, :] = TF.rotate(
                    x[i, :, :],
                    angle,
                    expand=self.expand,
                    center=self.center,
                    fill=self.fill,
                )
        else:
            raise ValueError("Input must be a 3D or 4D tensor.")
        return x
