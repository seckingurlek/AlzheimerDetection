import torch
import torch.nn as nn
from torchvision import models
from Preprocessing import preprocess_image
from Dataset_Loader import load_data

_, _, classes = load_data()

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, len(classes))
model.load_state_dict(torch.load("models/alzheimer_model.pth"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(image_path):
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]

# Örnek tahmin
image_path = "data/raw/sample_mri.jpg"
print(f"Tahmin: {predict(image_path)}")
