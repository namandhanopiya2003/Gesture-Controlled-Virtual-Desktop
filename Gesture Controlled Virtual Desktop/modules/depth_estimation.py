# Importing necessary libraries
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Class to estimate depth from a single RGB frame using a pre-trained model
class DepthEstimator:
    def __init__(self, model_path="models/depth_model/"):
        # Loads the pre-trained model
        self.model = torch.hub.load("nianticlabs/monodepth2", "mono_1024x320", pretrained=True).eval()
        # Defines preprocessing transformations: resize image and conversion
        self.transform = transforms.Compose([transforms.Resize((320,1024)), transforms.ToTensor()])

    def estimate(self, frame: np.ndarray) -> float:
        # Converts the frame to Image
        img = Image.fromarray(frame)
        # Applies preprocessing transformations and adds a batch dimension
        input_tensor = self.transform(img).unsqueeze(0)
        # Runs forward pass through the model without computing gradients
        with torch.no_grad():
            disp = self.model(input_tensor)[("disp", 0)]
        # Converts disparity to depth and moves array
        depth = 1 / disp.squeeze().cpu().numpy()
        # Returns the median depth value
        return np.median(depth)
