import argparse
from src.Dataset_Loader import prepare_data, load_data
from src.Model import AlzheimerCNN
from src.Train import train_model
from src.Evaluate import evaluate_model
from src.Inference import predict_image
import torch

class MainController:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 16
        self.model_path = "models/alzheimer_cnn.pth"

    def prepare_dataset(self):
        """Dataset'i train ve test olarak ayırır."""
        print("🔄 Dataset hazırlanıyor...")
        prepare_data()
        print("✅ Dataset hazır!")

    def train(self, epochs=10):
        """Modeli eğitir ve kaydeder."""
        print("🚀 Model eğitimi başlıyor...")
        train_loader, test_loader, classes = load_data(self.batch_size)
        model = AlzheimerCNN(len(classes)).to(self.device)
        train_model(model, train_loader, epochs, self.model_path)
        print("✅ Model eğitildi ve kaydedildi!")

    def evaluate(self):
        """Modelin test seti üzerindeki performansını ölçer."""
        print("📊 Model değerlendirmesi yapılıyor...")
        _, test_loader, classes = load_data(self.batch_size)
        model = AlzheimerCNN(len(classes)).to(self.device)
        evaluate_model(model, test_loader, self.model_path)
        print("✅ Değerlendirme tamamlandı!")

    def infer(self, image_path):
        """Yeni bir MRI görüntüsü ile tahmin yapar."""
        print(f"🔍 Tahmin yapılıyor: {image_path}")
        model = AlzheimerCNN(4).to(self.device)
        result = predict_image(model, image_path, self.model_path)
        print(f"🧠 Tahmin sonucu: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alzheimer Teşhis Sistemi")
    parser.add_argument("--prepare_data", action="store_true", help="Dataset'i hazırla")
    parser.add_argument("--train", action="store_true", help="Modeli eğit")
    parser.add_argument("--evaluate", action="store_true", help="Modeli test et")
    parser.add_argument("--infer", type=str, help="Bir MRI görüntüsü üzerinde tahmin yap")

    args = parser.parse_args()
    controller = MainController()

    if args.prepare_data:
        controller.prepare_dataset()
    elif args.train:
        controller.train()
    elif args.evaluate:
        controller.evaluate()
    elif args.infer:
        controller.infer(args.infer)
    else:
        print("❌ Geçersiz seçenek! --help ile komutları görebilirsin.")
