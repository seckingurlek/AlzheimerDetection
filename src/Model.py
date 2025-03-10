import torch
import torch.nn as nn
import torch.nn.functional as F

class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes=4):
        """
        Alzheimer MRI görüntülerini sınıflandıran CNN modeli.

        :param num_classes: Kaç farklı Alzheimer seviyesi olduğu (Varsayılan: 4)
        """
        super(AlzheimerCNN, self).__init__()

        # Convolutional Katmanları
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully Connected (Dense) Katmanları
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # 224x224 giriş boyutuna göre hesaplandı
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # Dropout (Overfitting önlemek için)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """İleri yayılım (Forward Propagation)"""
        x = F.relu(self.bn1(self.conv1(x)))  # 1. Konvolüsyon + BatchNorm + ReLU
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 1. Max Pooling

        x = F.relu(self.bn2(self.conv2(x)))  # 2. Konvolüsyon + BatchNorm + ReLU
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 2. Max Pooling

        x = F.relu(self.bn3(self.conv3(x)))  # 3. Konvolüsyon + BatchNorm + ReLU
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 3. Max Pooling

        # Flatten (Tam bağlantılı katmanlara giriş için düzleştirme)
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))  # 1. Fully Connected Katman
        x = self.dropout(x)  # Dropout ekleyerek overfitting önlenir
        x = F.relu(self.fc2(x))  # 2. Fully Connected Katman
        x = self.fc3(x)  # Çıkış Katmanı

        return x

# Test amaçlı model oluşturma
if __name__ == "__main__":
    model = AlzheimerCNN(num_classes=4)
    print(model)
