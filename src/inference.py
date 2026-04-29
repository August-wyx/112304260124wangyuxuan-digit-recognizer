import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .model import EnhancedCNN


class DigitRecognizer:
    def __init__(self, model_path='models/model.pth'):
        self.model = EnhancedCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        self.model.eval()

    def predict_uploaded_image(self, image):
        img = image.convert('L')
        img = Image.eval(img, lambda x: 255 - x)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_np = np.array(img) / 255.0
        tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()

        prediction = int(torch.argmax(output, dim=1).item())
        confidence = probabilities[prediction]

        top3_indices = np.argsort(probabilities)[::-1][:3]
        top3 = [{'digit': int(i), 'probability': probabilities[i]} for i in top3_indices]

        all_probabilities = [{'digit': i, 'probability': probabilities[i]} for i in range(10)]

        return {
            'prediction': prediction,
            'confidence': confidence,
            'top3': top3,
            'probabilities': all_probabilities
        }

    def predict_canvas(self, data):
        image_data = data.get('composite', data)
        img = Image.fromarray(image_data.astype('uint8'), mode='RGBA')
        img = img.convert('L')
        img = Image.eval(img, lambda x: 255 - x)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_np = np.array(img) / 255.0
        tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().tolist()

        prediction = int(torch.argmax(output, dim=1).item())
        confidence = probabilities[prediction]

        top3_indices = np.argsort(probabilities)[::-1][:3]
        top3 = [{'digit': int(i), 'probability': probabilities[i]} for i in top3_indices]

        all_probabilities = [{'digit': i, 'probability': probabilities[i]} for i in range(10)]

        return {
            'prediction': prediction,
            'confidence': confidence,
            'top3': top3,
            'probabilities': all_probabilities
        }
