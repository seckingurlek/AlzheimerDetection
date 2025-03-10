import os
import shutil
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Orijinal dataset konumu
DATASET_PATH = r"Users\90534\OneDrive\Masaüstü\Alzheimer_s Dataset"

# Train ve Test klasörleri
DESTINATION_PATH = "data"
TRAIN_RATIO = 0.8  # %80 train, %20 test

# Klasör eşleşmeleri
CATEGORY_MAPPING = {
    "VeryMildDemented": "VeryMildDemented",
    "MildDemented": "MildDemented",
    "ModerateDemented": "ModerateDemented",
    "NonDemented": "NonDemented"
}

def prepare_data():
    """Dataset'i train/test olarak ayırıp ilgili klasörlere kopyalar."""
    for label, source_folder in CATEGORY_MAPPING.items():
        source_path = os.path.join(DATASET_PATH, source_folder)
        train_dest = os.path.join(DESTINATION_PATH, "train", label)
        test_dest = os.path.join(DESTINATION_PATH, "test", label)

        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(test_dest, exist_ok=True)

        images = os.listdir(source_path)
        random.shuffle(images)

        train_size = int(len(images) * TRAIN_RATIO)
        train_images, test_images = images[:train_size], images[train_size:]

        for img in train_images:
            shutil.copy(os.path.join(source_path, img), os.path.join(train_dest, img))
        for img in test_images:
            shutil.copy(os.path.join(source_path, img), os.path.join(test_dest, img))

        print(f"{label}: {len(train_images)} train, {len(test_images)} test örneği ayrıldı.")

def load_data(batch_size=16):
    """Hazırlanan dataset'ten train ve test verisini yükler."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(DESTINATION_PATH, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(DESTINATION_PATH, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    prepare_data()  # Dataset'i hazırla
    print("Dataset başarıyla train/test olarak ayrıldı.")
