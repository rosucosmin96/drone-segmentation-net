import os
import torch

from utils import pixel_accuracy, mIoU
from config import config


def train(epochs, model, train_loader, val_loader, criterion, optimizer):
    checkpoint_path = os.path.join(r"./checkpoints", config["experiment"])
    if os.path.isdir(checkpoint_path) is False:
        os.mkdir(checkpoint_path)

    print("Training Started...")
    train_losses = []
    train_accs = []
    train_ious = []
    val_losses = []
    val_accs = []
    val_ious = []

    len_loader = len(train_loader)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        train_iou = 0

        for imgs, masks in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += pixel_accuracy(outputs, masks)
            train_iou += mIoU(outputs, masks)

        train_loss /= len_loader
        train_acc /= len_loader
        train_iou /= len_loader

        if epoch % 3 == 0:
            val_loss, val_acc, val_iou = evaluate(model, val_loader, criterion)

            print("Epoch: {}/{} | T loss: {:.4f} | T Acc: {:.2f} | T IoU: {:.2f} | V loss: {:.4f} | V Acc: {:.2f} | V "
                  "IoU: {:.2f}".format(epoch, epochs, train_loss, train_acc, train_iou, val_loss, val_acc, val_iou))

            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_ious.append(val_iou)

            torch.save(model.state_dict(), os.path.join(checkpoint_path, "{}_state_dict_{}.pth".format(config["backbone"], epoch)))

        else:
            print("Epoch: {}/{} | T loss: {:.4f} | T Acc: {:.4f} | T IoU: {:.2f}".format(epoch, epochs, train_loss,
                                                                                         train_acc, train_iou))
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_ious.append(train_iou)

    history = {'train_loss': train_losses, 'val_loss': val_losses,
               'train_miou': train_ious, 'val_miou': val_ious,
               'train_acc': train_accs, 'val_acc': val_accs}

    torch.save(model.state_dict(), os.path.join(checkpoint_path, "{}_state_dict.pth".format(config["backbone"])))

    return model, history


def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    val_acc = 0
    val_iou = 0

    len_loader = len(val_loader)

    with torch.no_grad():
        for imgs, masks in val_loader:
            outputs = model(imgs)
            val_loss += criterion(outputs, masks)
            val_acc += pixel_accuracy(outputs, masks)
            val_iou += mIoU(outputs, masks)

        return (
                val_loss / len_loader,
                val_acc / len_loader,
                val_iou / len_loader
                )


def binary_train(epochs, model, train_loader, val_loader, criterion, optimizer):
    checkpoint_path = os.path.join(r"./checkpoints", config["experiment"])
    if os.path.isdir(checkpoint_path) is False:
        os.mkdir(checkpoint_path)

    print("Training Started...")
    train_losses = []
    train_accs = []
    train_ious = []
    val_losses = []
    val_accs = []
    val_ious = []

    len_loader = len(train_loader)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        train_iou = 0

        for imgs, masks in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += pixel_accuracy(outputs, masks, binary=True)
            train_iou += mIoU(outputs, masks, n_classes=1, binary=True)

        train_loss /= len_loader
        train_acc /= len_loader
        train_iou /= len_loader

        if epoch % 3 == 0:
            val_loss, val_acc, val_iou = binary_evaluate(model, val_loader, criterion)

            print("Epoch: {}/{} | T loss: {:.4f} | T Acc: {:.2f} | T IoU: {:.2f} | V loss: {:.4f} | V Acc: {:.2f} | V "
                  "IoU: {:.2f}".format(epoch, epochs, train_loss, train_acc, train_iou, val_loss, val_acc, val_iou))

            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_ious.append(val_iou)

            torch.save(model.state_dict(), os.path.join(checkpoint_path, "{}_state_dict_{}.pth".format(config["backbone"], epoch)))

        else:
            print("Epoch: {}/{} | T loss: {:.4f} | T Acc: {:.4f} | T IoU: {:.2f}".format(epoch, epochs, train_loss,
                                                                                         train_acc, train_iou))
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_ious.append(train_iou)

    history = {'train_loss': train_losses, 'val_loss': val_losses,
               'train_miou': train_ious, 'val_miou': val_ious,
               'train_acc': train_accs, 'val_acc': val_accs}

    torch.save(model.state_dict(), os.path.join(checkpoint_path, "{}_state_dict.pth".format(config["backbone"])))

    return model, history


def binary_evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    val_acc = 0
    val_iou = 0

    len_loader = len(val_loader)

    with torch.no_grad():
        for imgs, masks in val_loader:
            outputs = model(imgs)
            val_loss += criterion(outputs, masks)
            val_acc += pixel_accuracy(outputs, masks, binary=True)
            val_iou += mIoU(outputs, masks, n_classes=1, binary=True)

        return (
                val_loss / len_loader,
                val_acc / len_loader,
                val_iou / len_loader
                )










