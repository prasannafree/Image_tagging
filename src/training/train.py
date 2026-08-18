import os 
import sys 
import yaml 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm 



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import WarehouseTaggingDataset, get_transforms
from models.tagging_model import (
    ResNet50TaggingNN,
    MobileNetV3TaggingNN,
    EfficientNetB0TaggingNN,
    CustomTaggingNN )

def load_config(config_path="src/config.yaml"):
    with open(config_path ,'r') as file:
        config = yaml.safe_load(file)
    return config 

def get_model(model_name , num_classes):
    if model_name == "resnet50":
        return ResNet50TaggingNN(num_classes=num_classes)
    elif model_name == "mobilenet_v3":
        return MobileNetV3TaggingNN(num_classes=num_classes)
    elif model_name == "efficientnet_b0":
        return EfficientNetB0TaggingNN(num_classes=num_classes)
    elif model_name == "custom":
        return CustomTaggingNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

def train():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    writer = SummaryWriter(log_dir=config['logging']['tensorboard_dir'])
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)

    train_transform = get_transforms(is_train = True) # tranforming the train data 
    val_transform = get_transforms(is_train = False) # trandforming the val data

    train_dataset = WarehouseTaggingDataset(
        csv_file=config['data']['train_csv'],
        img_dir=config['data']['train_img_dir'],
        transform=train_transform)

    val_dataset = WarehouseTaggingDataset(
        csv_file=config['data']['val_csv'],
        img_dir=config['data']['val_img_dir'],
        transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['training']['num_workers'])
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['training']['num_workers'])
    
    num_classes = train_dataset.num_classes
    model = get_model(config['model']['architecture'], num_classes).to(device)
    
    criterion = nn.BCEWithLogitsLoss() # we are using binary cross-entropy with logits loss as the model is used for multi class classification
    optimizer = optim.AdamW(model.parameters(), lr = config['training']['learning_rate'] , weight_decay = config['training']['weight_decay'])
    
    epochs = config['training']['epochs']
    best_val_loss = float('inf') # base loss value 
    checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model.pth') # storing the best model path 

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            # Move data to GPU
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)   # calculates the losses from the forward pass
            
            # Backward and optimize
            optimizer.zero_grad()    # clears out the math from previous batch
            loss.backward()          # do the calculus
            optimizer.step()         # updates the weights (the learning happens here)
            
            # Accumulate loss
            train_loss += loss.item()
            
            # Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())
            
        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        
        # 8. Validation Loop
        model.eval()              # prepare the NN for inference ( turning off layers like dropout and batch norm )
        val_loss = 0.0            # initialize the loss
        with torch.no_grad():     # tells the GPU to not to update the weights (no backprop)
            for images, labels in val_loader:
                images = images.to(device)   # sending images to GPU
                labels = labels.to(device)   # sending labels to GPU
                
                outputs = model(images)             # forward pass for validation
                loss = criterion(outputs, labels)   # calculates the losses from the forward pass
                val_loss += loss.item()             # accumulate the loss
                
        avg_val_loss = val_loss / len(val_loader)    # calculate the average loss
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)   # log the validation loss

        print(f"--> Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


        # 9. Model Checkpointing (Save the best model based on validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"🌟 Saved new best model to {checkpoint_path} with Val Loss: {best_val_loss:.4f}")
            
    writer.close()
    print("✅ Training Complete!")
    
    # ---------------------------------------------------------
    # 10. Test Evaluation (After training completes)
    # ---------------------------------------------------------
    print("\n🚀 Starting Test Evaluation on best model...")
    
    # Load the best model weights we saved during validation
    model.load_state_dict(torch.load(checkpoint_path, weights_only=False)['model_state_dict'])
    model.eval()
    test_dataset = WarehouseTaggingDataset(
        csv_file=config['data']['test_csv'],
        img_dir=config['data']['test_img_dir'],
        transform=val_transform  # We reuse val_transform because test data shouldn't be augmented
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['training']['num_workers'])
        
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
    avg_test_loss = test_loss / len(test_loader)
    print(f"🏆 Final Test Loss: {avg_test_loss:.4f}")

if __name__ == '__main__':
    train()
