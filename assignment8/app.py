import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path

# Import our model architecture
from model import ResNet18, create_model
from config import ModelConfig

# CIFAR-100 class names
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

class CIFAR100Classifier:
    def __init__(self, model_path="best_model.pth"):
        self.device = torch.device("cpu")  # HuggingFace Spaces uses CPU
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        
    def load_model(self, model_path):
        """Load the trained ResNet-18 model"""
        try:
            # Create model with CIFAR-100 configuration
            config = ModelConfig()
            model = create_model(config)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"Model loaded successfully from {model_path}")
            print(f"Best accuracy: {checkpoint.get('best_acc', 'Unknown')}%")
            
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def get_transform(self):
        """Get the preprocessing transforms for CIFAR-100"""
        return transforms.Compose([
            transforms.Resize((32, 32)),  # CIFAR-100 size
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    
    def predict(self, image):
        """Make prediction on input image"""
        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 5 predictions
            top5_prob, top5_idx = torch.topk(probabilities, 5)
            
            results = {}
            for i in range(5):
                class_idx = top5_idx[i].item()  # Ensure it's a Python int
                class_name = CIFAR100_CLASSES[class_idx]
                confidence = float(top5_prob[i].item())  # Keep as decimal (0-1) for Gradio
                # Ensure reasonable confidence values (0-1)
                confidence = max(0.0, min(1.0, confidence))
                results[class_name] = round(confidence, 4)  # Round to 4 decimal places
            
            return results
            
        except Exception as e:
            return {"Error": f"Prediction failed: {str(e)}"}

# Initialize the classifier
try:
    classifier = CIFAR100Classifier()
    model_loaded = True
except Exception as e:
    print(f"Failed to load model: {e}")
    model_loaded = False

def classify_image(image):
    """Gradio interface function"""
    if not model_loaded:
        return {"Error": "Model not loaded properly"}
    
    if image is None:
        return {"Error": "Please upload an image"}
    
    return classifier.predict(image)

# Create Gradio interface
def create_interface():
    title = "🎯 ResNet-18 CIFAR-100 Classifier"
    description = """
    ## High-Performance Image Classification
    
    This model is a **ResNet-18** trained from scratch on **CIFAR-100** dataset, achieving **74.07% test accuracy**.
    
    ### 📊 Model Performance:
    - **Architecture**: ResNet-18 (11.2M parameters)
    - **Dataset**: CIFAR-100 (100 classes)
    - **Test Accuracy**: 74.07%
    - **Training Time**: 115.5 minutes on Apple Silicon MPS
    - **Training Epochs**: 86 epochs
    
    ### 🚀 How to use:
    1. Upload any image (will be resized to 32x32)
    2. Get top-5 predictions with confidence scores
    3. The model works best with objects similar to CIFAR-100 classes
    
    ### 📋 CIFAR-100 Classes Include:
    Animals, vehicles, household items, plants, food, and more!
    """
    
    article = """
    ### 🔬 Technical Details:
    - **Model**: Custom ResNet-18 implementation
    - **Training**: SGD optimizer with CosineAnnealingLR scheduler
    - **Data Augmentation**: RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
    - **Device**: Trained on Apple Silicon MPS, deployed on CPU
    - **Framework**: PyTorch
    
    ### 📈 Training Progress:
    - Started at 12.53% accuracy (epoch 1)
    - Reached 73% target at epoch 81
    - Continued training to 74.07% at epoch 86
    - Excellent generalization: 97.89% train vs 74.07% test accuracy
    
    Built with ❤️ using PyTorch and Gradio
    """
    
    examples = [
        ["example_tiger.png"],
        ["example_rose.png"],
        ["example_bicycle.png"],
        ["example_dolphin.png"],
        ["example_rocket.png"],
    ]
    
    interface = gr.Interface(
        fn=classify_image,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
        title=title,
        description=description,
        article=article,
        examples=examples,
        theme=gr.themes.Soft(),
        analytics_enabled=False,
    )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch()
