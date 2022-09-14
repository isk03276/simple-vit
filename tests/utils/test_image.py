import sys
import unittest

import torch

try:
    from utils.image import slice_image_to_patches
except ModuleNotFoundError:
    sys.path.append(".")
    from utils.image import slice_image_to_patches


class TestImage(unittest.TestCase):
    def test_slice_image_to_patches(self):
        images = torch.zeros((10, 3, 200, 200))
        patch_size = 2
        patches = slice_image_to_patches(images, patch_size)
        assert patches.shape == (10, 10000, 3 * patch_size * patch_size)


if __name__ == "__main__":
    unittest.main()
