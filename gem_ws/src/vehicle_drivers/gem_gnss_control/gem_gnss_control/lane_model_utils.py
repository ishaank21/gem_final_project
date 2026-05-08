import os
import torch
import numpy as np
import cv2
from gem_gnss_control.simple_enet import SimpleENet


def load_model(data_dir: str) -> SimpleENet:
    path = os.path.join(data_dir, "data", "checkpoints", "epoch60.pth")
    model = SimpleENet()
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def inference(model: SimpleENet, image: np.ndarray, device) -> np.ndarray:
    image_height = image.shape[0]
    image_width = image.shape[1]

    resized_image = cv2.resize(image, (640, 384))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        yp = model(tensor)
        cls = torch.argmax(yp, dim=1)

    mask = cls.squeeze(0).cpu().numpy().astype(np.uint8)
    return cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
