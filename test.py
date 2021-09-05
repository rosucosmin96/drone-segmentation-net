import torch
import torch.nn.functional as F

from utils import save_maskedImages
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


