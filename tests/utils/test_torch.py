import shutil
import sys
import unittest

import torch

try:
    from vision_transformer.layers import MLPBlock
    from utils.torch import save_model, load_model
except ModuleNotFoundError:
    sys.path.append(".")
    from vision_transformer.layers import MLPBlock
    from utils.torch import save_model, load_model


class TestTorch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_model1 = MLPBlock(1)
        self.test_model2 = MLPBlock(1)
        self.test_input = torch.ones(1)
        self.dir_name = "test_dir"
        self.file_name = "test"

    def setUp(self):
        super().setUp()
        save_model(self.test_model1, self.dir_name, self.file_name)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.dir_name)

    def test_save_and_load_model(self):
        load_model(self.test_model2, "test_dir/test")
        assert self.test_model1(self.test_input) == self.test_model2(self.test_input)


if __name__ == "__main__":
    unittest.main()
