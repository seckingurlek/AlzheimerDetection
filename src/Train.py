import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from Dataset_Loader import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader, classes = load_data()

# ResNet modelini kullan
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, len(classes))  # 4 sınıf: healthy, early, moderate, advanced
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/alzheimer_model.pth")
print("Model başarıyla kaydedildi.")
