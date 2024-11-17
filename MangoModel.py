import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
from torchvision import models, transforms
import os

class MangoModel(nn.Module):
    def __init__(self, num_classes_list):
        """
        Args:
            num_classes_list (list): List of integers, where each integer is the
            number of classes for an attribute.
        """
        super(MangoModel, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.shared_fc = nn.Sequential(*list(self.backbone.children())[:-1])  # Exclude final FC
        self.fcs = nn.ModuleList([
                nn.Sequential(
                nn.Linear(2048, 512),       # Intermediate layer
                nn.ReLU(),                  # Non-linear activation
                nn.Dropout(0.5),            # Regularization
                nn.Linear(512, num_classes) # Output layer
            ) for num_classes in num_classes_list
        ])

    def forward(self, x):
        x = self.shared_fc(x)
        x = torch.flatten(x, 1)
        outputs = [fc(x) for fc in self.fcs]
        return outputs
    
def main():
    # model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to('cuda')
    mangoModel = MangoModel([10, 5, 3]).to('cuda')
    mangoTransforms = ResNet50_Weights.IMAGENET1K_V1.transforms
    
    mangoTransforms = transforms.Compose([
        transforms.Resize(224),               # Resize the shorter side to 256
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
        
    from MangoData import MangoData
    
    img_folder = os.path.join('images', 'images')
    
    mangoData = MangoData('train.csv', img_folder = img_folder, additional_transform = mangoTransforms)
    img, label = mangoData[0]
    
    img = img.unsqueeze(0).to('cuda')
    output = mangoModel(img)
    
    print(output)
    
if __name__ == '__main__':
    main()
    
