import os
import argparse

import datetime

import numpy as np

from dataset.dataset_getter import DatasetGetter
from vision_transformer.models import ViT
from vision_transformer.learner import ViTLearner
from utils.torch import get_device, save_model, load_model
from utils.log import TensorboardLogger
from utils.config import save_yaml, load_from_yaml


def get_current_time() -> str:
    """
    Generate current time as string.
    Returns:
        str: current time
    """
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")
    return curr_time


def run(args):
    device = get_device(args.device)

    # Getting Dataset
    dataset = DatasetGetter.get_dataset(
        dataset_name=args.dataset_name, path=args.dataset_path, is_train=not args.test
    )
    dataset_loader = DatasetGetter.get_dataset_loader(
        dataset=dataset, batch_size=1 if args.test else args.batch_size
    )
    sampled_data = next(iter(dataset_loader))[0]
    n_channel, image_size = sampled_data.size()[1:3]

    # Model Instantiation
    if args.load_from and args.load_model_config:
        dir_path = os.path.dirname(args.load_from)
        config_file_path = dir_path + "/config.yaml"
        config = load_from_yaml(config_file_path)
        args.patch_size = config["patch_size"]
        args.embedding_size = config["embedding_size"]
        args.encoder_blocks_num = config["encoder_blocks_num"]
        args.heads_num = config["heads_num"]
        args.classes_num = config["classes_num"]

    model = ViT(
        image_size=image_size,
        n_channel=n_channel,
        n_patch=args.patch_size,
        n_dim=args.embedding_size,
        n_encoder_blocks=args.encoder_blocks_num,
        n_heads=args.heads_num,
        n_classes=args.classes_num,
    ).to(device)
    if args.load_from is not None:
        load_model(model, args.load_from)

    # Train / Test Iteration
    learner = ViTLearner(model=model)
    epoch = 1 if args.test else args.epoch

    if not args.test:
        model_save_dir = "{}/{}/".format(args.save_dir, get_current_time())
        logger = TensorboardLogger(model_save_dir)
        logger.add_model_graph(model=model, image=sampled_data)
        save_yaml(vars(args), model_save_dir + "config.yaml")

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
        loss_avg, acc_avg = np.mean(loss_list), np.mean(acc_list)
        if not args.test:
            # Save model
            if (epoch + 1) % args.save_interval == 0:
                save_model(model, model_save_dir, "epoch_{}".format(epoch + 1))
            # Log
            logger.log(tag="Training/Loss", value=loss_avg, step=epoch + 1)
            logger.log(tag="Training/Accuracy", value=acc_avg, step=epoch + 1)

        print("[Epoch {}] Loss : {} | Accuracy : {}".format(epoch, loss_avg, acc_avg))
        
    logger.close()


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
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    # save / load
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints/", help="Dataset name (ex. cifar10"
    )
    parser.add_argument(
        "--save-interval", type=int, default=5, help="Model save interval"
    )
    parser.add_argument("--load-from", type=str, help="Path to load the model")
    parser.add_argument(
        "--load-model-config",
        action="store_true",
        help="Whether to use the config file of the model to be loaded",
    )

    args = parser.parse_args()
    run(args)
