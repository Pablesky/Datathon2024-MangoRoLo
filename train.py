import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from MangoModel import MangoModel
from MangoData import MangoData
import os
from tqdm import tqdm
import warnings

# Suppress the specific UserWarning for the missing image functionality
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = [target.to(device) for target in targets]
        
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        
        # Compute the loss for each output (since you have multiple outputs)
        loss = sum(criterion(out, tgt.argmax(dim=1)) for out, tgt in zip(outputs, targets))
        
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters
        running_loss += loss.item() * inputs.size(0)
    
    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss

def main():
    # Model Definition
    numClasses = [7, 7, 12, 6, 13, 34, 34, 7, 5, 5, 5]
    mangoModel = MangoModel(num_classes_list=numClasses).to(device)
    
    # Data Preparation
    mangoTransforms = transforms.Compose([
        transforms.Resize(224),               # Resize the shorter side to 224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img_path = os.path.join('images', 'images')
    mangoDataset = MangoData(
        csv_path='train.csv', 
        img_folder=img_path, 
        additional_transform=mangoTransforms)
    
    mangoDataloader = DataLoader(
        mangoDataset, 
        batch_size=8, 
        shuffle=True,
        num_workers=os.cpu_count()
    )
    
    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()  # For each output in your model
    optimizer = optim.Adam(mangoModel.parameters(), lr=1e-4)
    
    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        train_loss = train_one_epoch(mangoModel, mangoDataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        # If you have a validation set, you can add evaluation here
        # val_loss, val_accuracy = validate_one_epoch(mangoModel, val_dataloader, criterion, device)
        # print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # Save the model
        torch.save(mangoModel.state_dict(), "mango_model.pth")

if __name__ == '__main__':
    main()
