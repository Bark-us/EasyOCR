import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict

from easyocr.craft import CRAFT
from easyocr.model import Model


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def get_detector(trained_model, device='cpu'):
    net = CRAFT()

    if device == 'cpu':
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = False

    net.eval()
    return net


def save_detector(model_in, model_out):
    detector = get_detector(model_in)
    torch.save(detector.state_dict(), model_out)


def save_recognizer(model_in, model_out):
    recognizer = Model(1, 512, 512, 168)
    state_dict = torch.load(model_in, map_location='cpu')
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:]
        new_state_dict[new_key] = value
    recognizer.load_state_dict(new_state_dict)
    torch.save(recognizer.state_dict(), model_out)


if __name__ == '__main__':
    save_detector('../model/craft_mlt_25k.pth', 'compat_models/craft.pth')
    save_recognizer('../model/latin.pth', 'compat_models/text.pth')