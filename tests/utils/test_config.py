import os
import sys
import unittest
import yaml

try:
    from utils.config import load_from_yaml, save_yaml
except ModuleNotFoundError:
    sys.path.append(".")
    from utils.config import load_from_yaml, save_yaml


class TestBaseParser(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_dict = {"a": 123, "b": "test"}
        self.file_path = "./test.yaml"

    def setUp(self):
        super().setUp()
        save_yaml(dict_data=self.config_dict, yaml_file_path=self.file_path)

    def tearDown(self):
        super().tearDown()
        os.remove(self.file_path)

    def test_load_from_yaml(self):
        yaml_config = load_from_yaml(self.file_path)
        for key, value in self.config_dict.items():
            assert value == yaml_config[key]


if __name__ == "__main__":
    unittest.main()
