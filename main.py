import torch
import cv2
import torch.nn as nn
import albumentations as A
import segmentation_models_pytorch as smp

from torch.optim import Adam

from config import config
from utils import create_df, split_dataset, plot_loss, plot_acc, plot_iou
from dataset import get_loader
from train import train
from test import test, test_image


def main(train_mode=True, load_model=None):
    # Set device according to the hardware
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training device: ", device)

    df = create_df()
    X_train, X_val, X_test = split_dataset(df)
    print("Datasets created...")

    t_train = A.Compose([A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(),
                         A.GaussNoise(), A.RandomBrightnessContrast((0, 0.5), (0, 0.5))])
    t_val = A.Compose([A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip()])

    train_loader = get_loader(config["IMAGE_DIR"], config["MASK_DIR"], X_train, mean=config["mean"], std=config["std"],
                              transform=t_train, batch_size=config["batch_size"], device=device)
    val_loader = get_loader(config["IMAGE_DIR"], config["MASK_DIR"], X_val, mean=config["mean"], std=config["std"],
                            transform=t_val, batch_size=config["batch_size"], device=device)

    model = smp.Unet(config["backbone"], encoder_weights='imagenet', classes=config["num_classes"], activation=None,
                     encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16]).to(device)
    print("Model created...")

    if load_model:
        model.load_state_dict(torch.load(load_model, map_location=torch.device(device)))
        print("Model loaded...")

    if train_mode:
        optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        criterion = nn.CrossEntropyLoss()

        model, history = train(config["num_epochs"], model, train_loader, val_loader, criterion, optimizer)

        plot_loss(history)
        plot_acc(history)
        plot_iou(history)

    t_test = A.Resize(768, 1152, interpolation=cv2.INTER_NEAREST)
    test_loader = get_loader(config["IMAGE_DIR"], config["MASK_DIR"], X_test, mean=config["mean"], std=config["std"],
                             transform=t_test, batch_size=1, shuffle=False, device=device)
    test(model, test_loader)

    # test_image(model, r"./data/dataset/semantic_drone_dataset/original_images/042.jpg", t_test, device)


if __name__ == '__main__':
    model_path = r'./checkpoints/resnet18_state_dict.pth'
    main(train_mode=False, load_model=model_path)
