import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class DepthEstimator:
    def __init__(self, model_path="models/depth_model/"):
        self.model = torch.hub.load("nianticlabs/monodepth2", "mono_1024x320", pretrained=True).eval()
        self.transform = transforms.Compose([transforms.Resize((320,1024)), transforms.ToTensor()])

    def estimate(self, frame: np.ndarray) -> float:
        img = Image.fromarray(frame)
        input_tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            disp = self.model(input_tensor)[("disp", 0)]
        depth = 1 / disp.squeeze().cpu().numpy()
        return np.median(depth)
