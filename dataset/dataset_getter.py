import torchvision


class DatasetGetter:
    @staticmethod
    def get_dataset_cls(dataset_name: str):
        if dataset_name == "cifar10":
            return torchvision.datasets.CIFAR10
        elif dataset_name == "cifar100":
            return torchvision.datasets.CIFAR100
        else:
            raise NotImplementedError
    
    @staticmethod    
    def get_dataset(dataset_name: str = "cifar10", path: str = "data/", is_train: bool = True, download: bool = True, transform = None):
        dataset_cls = DatasetGetter.get_dataset_cls(dataset_name=dataset_name)
        dataset = dataset_cls(root=path, train=is_train, download=download, transform=transform)
        return dataset
    