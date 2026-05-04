import torch
import numpy as np
import cv2
from simple_enet import SimpleENet


##### YOUR CODE STARTS HERE #####
# DO NOT CHANGE ANY FUNCTION HEADERS

# load your best model
def load_model() -> SimpleENet:
    #path_to_your_model = "data/FILL_THIS_OUT"   # Best Path for now but we can update: /home/ark11/Documents/ece484_gn_ak_sp/mp1-sp26-abo/src/mp1/data/checkpoints
    path_to_your_model = "data/checkpoints/epoch60.pth"
    model = SimpleENet()
    model.load_state_dict(torch.load(path_to_your_model, weights_only=True))
    # model.eval()
    return model

def inference(model: SimpleENet, image: np.ndarray, device: str) -> np.ndarray:
    """
    The main image processing pipeline for your model
    
    :param model: pytorch model
    :type model: SimpleENet
    :param image: a BGR image taken from the GEM's camera
    :type image: np.ndarray
    :param device: the device on which the model should run on ("cpu" or "cuda")
    :type device: str
    :return: binary lane-segmented image
    :rtype: ndarray
    """
    #pred = None

    image_height = image.shape[0]
    image_width = image.shape[1]

    resized_image = cv2.resize(image, (640, 384))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    new_shape = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)

    dev = torch.device(device)
    model = model.to(dev)
    new_shape = new_shape.to(dev)

    # model.eval()

    with torch.no_grad():
        yp = model(new_shape)
        cls = torch.argmax(yp, dim=1)

    mask = (cls.squeeze(0).cpu().numpy().astype(np.uint8))
    pred = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)


    return pred


##### YOUR CODE ENDS HERE #####