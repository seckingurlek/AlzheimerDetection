import torch
from Dataset_Loader import load_data
from Train import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, test_loader, classes = load_data()

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Doğruluk: {accuracy:.2f}%")
