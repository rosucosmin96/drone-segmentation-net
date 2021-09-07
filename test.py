import torch
import cv2
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import transforms as T
from utils import save_maskedImages, blur_class
from config import config


def test(model, test_loader):
    model.eval()

    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(test_loader):
            outputs = model(imgs)
            output = outputs.squeeze()
            output = F.softmax(output, dim=0)
            output = torch.argmax(output, dim=0)

            mask = masks.squeeze()
            img = imgs.squeeze()

            save_maskedImages(img, output, mask, title="{}_{}".format(config["backbone"], idx))


def test_image(model, img_path, transform, device):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(image=img)["image"]

    t = T.Compose([T.ToTensor(), T.Normalize(config["mean"], config["std"])])
    tensor_img = t(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor_img)
        output = output.squeeze()
        output = torch.argmax(torch.softmax(output, dim=0), dim=0)

    mask = output.cpu().numpy()
    blurred_img, person_mask = blur_class(img, mask, 15, kernel=(20, 20))
    plt.subplot(1,3,1)
    plt.imshow(blurred_img)
    plt.subplot(1,3,2)
    plt.imshow(person_mask)
    plt.subplot(1,3,3)
    plt.imshow(img)
    plt.show()








