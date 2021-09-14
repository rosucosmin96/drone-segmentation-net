import os
import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from config import config
from PIL import Image


def create_df():
    name = []
    for dirname, _, filenames in os.walk(config["IMAGE_DIR"]):
        for filename in filenames:
            name.append(filename.split('.')[0])

    return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))


def split_dataset(df, val_size=0.15, test_size=0.1):
    X_trainval, X_test = train_test_split(df.values, test_size=test_size, random_state=19)
    X_train, X_val = train_test_split(X_trainval, test_size=val_size, random_state=19)

    return X_train, X_val, X_test


def plot_maskedImage(img, mask, alpha=0.6, title='Picture with mask applied'):
    plt.imshow(img)
    plt.imshow(mask, alpha=alpha)
    plt.title(title)
    plt.show()


def save_maskedImages(img, output, mask, alpha=0.6, title='Picture with mask applied'):
    save_path = os.path.join(r"./saved_images", config["experiment"])
    if os.path.isdir(save_path) is False:
        os.mkdir(save_path)

    n_img = np.transpose(img.cpu().numpy(), (1, 2, 0))

    plt.subplot(1, 2, 1)
    plt.imshow(n_img)
    plt.imshow(mask.cpu(), alpha=alpha)
    plt.title("Ground Truth")

    plt.subplot(1, 2, 2)
    plt.imshow(n_img)
    plt.imshow(output.cpu(), alpha=alpha)
    plt.title("Prediction")

    plt.savefig(save_path + "{}.jpg".format(title))


def pixel_accuracy(output, truth, binary=False):
    with torch.no_grad():
        if binary:
            output = torch.round(output)
        else:
            output = torch.argmax(F.softmax(output, dim=1), dim=1)

        correct = torch.eq(output, truth).int()
        acc = float(correct.sum()) / float(correct.numel())

    return acc * 100


def mIoU(output, mask, n_classes=23, binary=False):
    epsilon = 1e-8
    with torch.no_grad():
        if binary:
            output = torch.round(output)
        else:
            output = F.softmax(output, dim=1)
            output = torch.argmax(output, dim=1)

        output = output.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for categ in range(n_classes):
            true_class = output == categ
            true_label = mask == categ

            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + epsilon) / (union + epsilon) * 100
                iou_per_class.append(iou)

        return np.nanmean(iou_per_class)


def plot_loss(history):
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_iou(history):
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU', marker='*')
    plt.title('Score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def plot_acc(history):
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy', marker='*')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


def blur_class(img, mask, class_idx, kernel=(50, 50)):
    np_mask = np.array(mask)
    # np_mask[np.where(np_mask != class_idx)] = 0
    # np_mask[np.where(np_mask == class_idx)] = 1
    np_mask = np.isin(np_mask, class_idx)
    np_mask = np.stack((np_mask, np_mask, np_mask), axis=-1)

    np_img = np.array(img)

    # Taking the person from the image
    person_img = np_img * np_mask

    # Blur the whole image
    blurred_img = cv2.blur(np_img, kernel)

    # Extract the blurred person
    person_blur = blurred_img * np_mask
    # person_blur = cv2.blur(person_img, (50, 50))

    new_img = np_img - person_img + person_blur

    person_mask = Image.fromarray((np_mask * 255).astype(np.uint8))
    new_img = Image.fromarray(new_img.astype(np.uint8))

    return new_img, person_mask


if __name__ == '__main__':
    df = create_df()
    print(len(df))

    img = Image.open(config["IMAGE_DIR"] + df['id'][50] + '.jpg')
    mask = Image.open(config["MASK_DIR"] + df['id'][50] + '.png')
    plot_maskedImage(img, mask)

    new_img, person_mask = blur_class(img, mask, class_idx=[15, 17])
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(person_mask)
    plt.subplot(1, 3, 3)
    plt.imshow(new_img)
    plt.show()
