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
        h_window = 2
        w_window = 2
        patches = slice_image_to_patches(images, h_window, w_window)
        assert patches.shape == (10, 100, 100, 3, 2, 2)
        
        
if __name__ == "__main__":
    unittest.main()