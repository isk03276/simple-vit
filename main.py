import argparse

import numpy as np

from dataset.dataset_getter import DatasetGetter
from vision_transformer.models import ViT
from vision_transformer.learner import ViTLearner
from utils.torch import get_device


def run(args):
    device = get_device(args.device)

    dataset = DatasetGetter.get_dataset(
        dataset_name=args.dataset_name, path=args.dataset_path, is_train=not args.test
    )
    dataset_loader = DatasetGetter.get_dataset_loader(
        dataset=dataset, batch_size=1 if args.test else args.batch_size
    )
    n_channel, image_size = next(iter(dataset_loader))[0].size()[1:3]

    model = ViT(
        image_size=image_size,
        n_channel=n_channel,
        n_patch=args.patch_size,
        n_dim=args.embedding_size,
        n_encoder_blocks=args.encoder_blocks_num,
        n_heads=args.heads_num,
        n_classes=args.classes_num,
    ).to(device)
    learner = ViTLearner(model=model)

    epoch = 1 if args.test else args.epoch
    for epoch in range(epoch):
        loss_list, acc_list = [], []
        for images, labels in dataset_loader:
            images = images.to(device)
            labels = labels.to(device)
            loss, acc = learner.step(
                images=images, labels=labels, is_train=not args.test
            )
            loss_list.append(loss)
            acc_list.append(acc)
        print(
            "[Epoch {}] Loss : {} | Accuracy : {}".format(
                epoch, np.mean(loss_list), np.mean(acc_list)
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Transformer")
    # dataset
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device name to use GPU (ex. cpu, cuda, mps, etc.)",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="cifar10", help="Dataset name (ex. cifar10"
    )
    parser.add_argument(
        "--dataset-path", type=str, default="data/", help="Dataset path"
    )
    parser.add_argument("--classes-num", type=int, default=10, help="Number of classes")
    parser.add_argument("--patch-size", type=int, default=4, help="Image patch size")
    parser.add_argument(
        "--embedding-size", type=int, default=512, help="Number of hidden units"
    )
    parser.add_argument(
        "--encoder-blocks-num",
        type=int,
        default=8,
        help="Number of transformer encoder blocks",
    )
    parser.add_argument(
        "--heads-num", type=int, default=8, help="Number of attention heads"
    )
    # train / test
    parser.add_argument("--epoch", type=int, default=200, help="Learning epoch")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    # save / load
    parser.add_argument(
        "--save-interval", type=int, default=5, help="Model save interval"
    )
    parser.add_argument("--load-from", type=str, help="Path to load the model")

    args = parser.parse_args()
    run(args)
