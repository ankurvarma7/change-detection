import torch
import torchgeo
from torchgeo.datasets import OSCD
from torchvision.transforms import functional as TF
import random
from fc_ef import Unet
from fc_siam_diff import SiamUnet_diff
from fc_siam_conc import SiamUnet_conc
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        img1 = image["image1"]
        img2 = image["image2"]
        mask = image["mask"]
        return {
            "image1": img1,
            "image2": img2,
            "mask": mask
        }


def augment_image(image):
    """Apply random augmentations to the image."""
    augmented_images = []
    for _ in range(4):
        img1 = image["image1"]
        img2 = image["image2"]
        mask = image["mask"]

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        # Random horizontal flip
        if random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            mask = TF.hflip(mask)
        # Random vertical flip
        if random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            mask = TF.vflip(mask)
        # Random rotation
        angle = random.choice([0, 90, 180, 270])
        img1 = TF.rotate(img1, angle)
        img2 = TF.rotate(img2, angle)
        mask = TF.rotate(mask, angle)

        augmented_images.append({
            "image1": img1,
            "image2": img2,
            "mask": mask
        })
    return augmented_images

def get_model(model_type, input_nbr, label_nbr):
    """Get the model based on the specified type."""
    if model_type == "ef":
        return Unet(input_nbr, label_nbr)
    elif model_type == "siam_diff":
        return SiamUnet_diff(input_nbr, label_nbr)
    elif model_type == "siam_conc":
        return SiamUnet_conc(input_nbr, label_nbr)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
def train_model(model, model_type, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001):
    """Train the model."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_losses = []
    validation_losses = []
    for epoch in tqdm(range(num_epochs), desc=f"Training {model_type} model"):
        model.train()
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            img1 = batch["image1"]
            img2 = batch["image2"]
            mask = batch["mask"]

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            # Calculate metrics
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy().flatten()
            mask = mask.cpu().numpy().flatten()
            true_positive += ((predicted == 1) & (mask == 1)).sum()
            false_positive += ((predicted == 1) & (mask == 0)).sum()
            false_negative += ((predicted == 0) & (mask == 1)).sum()
            true_negative += ((predicted == 0) & (mask == 0)).sum()
        # Calculate precision, recall, and F1 score
        total = true_positive + false_positive + false_negative + true_negative
        if total == 0:
            print("No samples found in this epoch, skipping metrics calculation.")
            continue
        # Print metrics
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, Accuracy: {(true_positive + true_negative) / total:.4f}")
        # Print loss
        loss = loss / len(train_loader)  # Average loss over the batch

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss :.4f}")
        # Store training loss
        training_losses.append(loss)
        # Optionally, you can save the model state here
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"../model_state/model_epoch_{epoch+1}.pth")

        # Validation step can be added here
        model.eval()
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0
        with torch.no_grad():
            val_loss = 0.0
            for val_batch in val_loader:
                val_img1 = val_batch["image1"]
                val_img2 = val_batch["image2"]
                val_mask = val_batch["mask"]

                val_outputs = model(val_img1, val_img2)
                val_loss += criterion(val_outputs, val_mask).item()
                # Calculate metrics
                _, val_predicted = torch.max(val_outputs, 1)
                val_predicted = val_predicted.cpu().numpy().flatten()
                val_mask = val_mask.cpu().numpy().flatten()
                true_positive += ((val_predicted == 1) & (val_mask == 1)).sum()
                false_positive += ((val_predicted == 1) & (val_mask == 0)).sum()
                false_negative += ((val_predicted == 0) & (val_mask == 1)).sum()
                true_negative += ((val_predicted == 0) & (val_mask == 0)).sum()
            # Calculate precision, recall, and F1 score for validation
            val_total = true_positive + false_positive + false_negative + true_negative
            if val_total == 0:
                print("No samples found in validation, skipping metrics calculation.")
                continue
            val_precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            val_recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
            print(f"Validation - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1_score:.4f}, Accuracy: {(true_positive + true_negative) / val_total:.4f}")
            # Print validation loss
            print(f"Validation Loss: {val_loss / len(val_loader):.4f}")
            validation_losses.append(val_loss / len(val_loader))
        
        plt.plot(training_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'loss_plot_{model_type}.png')
        plt.clf()
        plt.close()
    
def test_model(model, test_loader, device):
    """Test the model."""
    model.eval()
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
        # Iterate over the test dataset
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            img1 = batch["image1"].to(device)
            img2 = batch["image2"].to(device)
            mask = batch["mask"].to(device)
            mask = mask.squeeze(1) if mask.dim() == 3 else mask  # Ensure mask is 2D

            outputs = model(img1, img2)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy().flatten()
            mask = mask.cpu().numpy().flatten()
            true_positive += ((predicted == 1) & (mask == 1)).sum()
            false_positive += ((predicted == 1) & (mask == 0)).sum()
            false_negative += ((predicted == 0) & (mask == 1)).sum()
            true_negative += ((predicted == 0) & (mask == 0)).sum()
    # Calculate precision, recall, and F1 score
    total = true_positive + false_positive + false_negative + true_negative
    if total == 0:
        print("No samples found in the test set, skipping metrics calculation.")
        return
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Test - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, Accuracy: {(true_positive + true_negative) / total:.4f}")
    

if __name__ == "__main__":
    train_dataset = OSCD(root="../data", split="train", download=True)
    test_dataset = OSCD(root="../data", split="test", download=True)


    augmented_train_dataset = []
    for image in train_dataset:
        augmented_images = augment_image(image)
        augmented_train_dataset.extend(augmented_images)

    train_dataset, val_dataset = torch.utils.data.random_split(
        augmented_train_dataset, [int(0.8 * len(augmented_train_dataset))+1, int(0.2 * len(augmented_train_dataset))]
    )
    train_loader = DataLoader(CustomDataset(train_dataset), batch_size=1, shuffle=True)
    val_loader = DataLoader(CustomDataset(val_dataset), batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_type = "siam_diff"  # Change to "ef" or "siam_conc" as needed
    input_nbr = 13  # Number of input channels
    label_nbr = 2  # Number of output classes
    model = get_model(model_type, input_nbr, label_nbr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_model(model, model_type, train_loader, val_loader, device=device, num_epochs=5, learning_rate=0.001)
    test_model(model, test_loader, device=device)