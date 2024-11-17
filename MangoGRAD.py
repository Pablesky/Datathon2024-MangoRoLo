from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import numpy as np
import cv2  # OpenCV for image manipulation
from MangoModel import MangoModel, SingleHeadMangoModel
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn


class SingleHeadMangoModel(nn.Module):
    def __init__(self, base_model, selected_head_idx):
        """
        Args:
            base_model (MangoModel): The original model with multiple heads.
            selected_head_idx (int): Index of the head to keep.
        """
        super(SingleHeadMangoModel, self).__init__()
        self.backbone = base_model.backbone
        self.shared_fc = base_model.shared_fc
        self.selected_fc = base_model.fcs[selected_head_idx]  # Select the desired head

    def forward(self, x):
        x = self.shared_fc(x)
        x = torch.flatten(x, 1)
        output = self.selected_fc(x)  # Forward through the selected head
        return output

img_path = r'C:\Users\ikerc\Desktop\Hackathon\archive\images\images'
file = '81_1064669_77044403-37_B.jpg'
numClasses = [7, 7, 12, 6, 13, 34, 34, 7, 5, 5, 5]
device = 'cpu'

# Initialize your MangoModel
model = MangoModel(num_classes_list=numClasses)
model.load_state_dict(torch.load('mango_model.pth', map_location=torch.device('cpu')))
# Create a single-head model
model = SingleHeadMangoModel(model, 6)
model.eval()

mangoTransforms = transforms.Compose([
    transforms.Resize((224, 224)),               # Resize the shorter side to 224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# Select target layers (assuming we want to apply Grad-CAM to the last layer of the backbone)
target_layers = [model.backbone.layer4[-1]]  # You can adjust this if needed

# Sample input tensor (replace with your actual input image tensor)
# input_tensor = torch.randn(1, 3, 224, 224)  # Example tensor, shape (batch_size, channels, height, width)
img = Image.open(os.path.join(img_path, file))
input_tensor = mangoTransforms(img)
# Add batch dimension and move the input image to the same device as the model
input_tensor = input_tensor.unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Specify which class output you want to target (e.g., the first head's class, class 281 for instance)
targets = [ClassifierOutputTarget(7)]  # Replace 281 with the target class for your head

with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Assume rgb_img is the original image in range [0, 1]
    rgb_img = mangoTransforms(img).numpy()
    rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
    rgb_img = rgb_img.astype(np.float32)
    rgb_img = rgb_img.transpose(1, 2, 0)

    # rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))  # Normalize to [0, 1]
    # rgb_img = rgb_img.astype(np.float32)  # Ensure dtype is np.float32  
    visualization = show_cam_on_image(rgb_img, cv2.resize(grayscale_cam, (224, 224)), use_rgb=True)
    
    cv2.imshow('Grad-CAM', visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()