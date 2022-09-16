from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


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
    def get_transform(normalize: bool = True, resize_shape: tuple = None):
        transform_compose_list = [transforms.ToTensor()]
        if normalize:
            transform_compose_list.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )
        if resize_shape:
            transform_compose_list.append(transforms.Resize(resize_shape))
        transform = transforms.Compose(transform_compose_list)
        return transform

    @staticmethod
    def get_dataset(
        dataset_name: str = "cifar10",
        path: str = "data/",
        is_train: bool = True,
        download: bool = True,
        transform=None,
    ) -> Dataset:
        dataset_cls = DatasetGetter.get_dataset_cls(dataset_name=dataset_name)
        if transform is None:
            transform = DatasetGetter.get_transform()
        dataset = dataset_cls(
            root=path, train=is_train, download=download, transform=transform
        )
        return dataset

    @staticmethod
    def get_dataset_loader(
        dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
