import torch
from models.u_net import UNet


def load_model(model_path: str, device) -> UNet:
    ckpt = torch.load(model_path, map_location=device)
    model = UNet(num_classes=1, in_channels=3).to(device)
    model.load_state_dict(state_dict=ckpt['model_state'])
    model.eval()
    print(f'model has been loaded | Best Dice: {ckpt["best_dice"]:.4f}')
    return model