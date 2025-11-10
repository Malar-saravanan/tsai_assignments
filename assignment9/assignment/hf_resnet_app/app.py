import os
from typing import List, Dict, Any, Union

import torch
from torch import nn
import torchvision.models as tvm
from torchvision.transforms import functional as F
from torchvision import transforms as T
from PIL import Image
import gradio as gr


CHECKPOINT_PATH = os.environ.get("CKPT_PATH", "best.pth")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes: int = 1000) -> nn.Module:
    model = tvm.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_preprocess_and_labels():
    # Use torchvision's ImageNet-1k metadata for categories and canonical transforms
    try:
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2
    except Exception:
        # Fallback if weights enum not available
        weights = None
    if weights is not None:
        preprocess = weights.transforms()
        labels = weights.meta.get("categories", [str(i) for i in range(1000)])
    else:
        preprocess = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        labels = [str(i) for i in range(1000)]
    return preprocess, labels


def load_checkpoint_into_model(model: nn.Module, checkpoint_path: str) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. "
            f"Place your file at runs/exp1/best.pth or set CKPT_PATH env var."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Support either a full training checkpoint dict or a raw state_dict
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()


device = get_device()
model = build_model(num_classes=1000).to(device)
preprocess, imagenet_labels = get_preprocess_and_labels()
load_checkpoint_into_model(model, CHECKPOINT_PATH)


def predict_images(
    images: Union[Image.Image, List[Image.Image]],
    top_k: int = 5,
) -> List[List[Dict[str, Any]]]:
    if images is None:
        return []
    if not isinstance(images, list):
        images = [images]

    results: List[List[Dict[str, Any]]] = []
    with torch.no_grad():
        for image in images:
            if not isinstance(image, Image.Image):
                # Some gradio versions may return dicts; handle defensively
                image = Image.fromarray(image)
            tensor = preprocess(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            topk = torch.topk(probs, k=top_k)
            sample_result: List[Dict[str, Any]] = []
            for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                label = imagenet_labels[idx] if 0 <= idx < len(imagenet_labels) else str(idx)
                sample_result.append({"label": label, "probability": float(score)})
            results.append(sample_result)
    return results


# Custom CSS for modern UI
custom_css = """
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
    max-width: 1400px !important;
}
.header-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 40px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}
.stats-card {
    background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #667eea;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.prediction-box {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
"""

with gr.Blocks(title="ResNet-50 ImageNet-1k Classifier", css=custom_css, theme=gr.themes.Soft()) as demo:
    # Header
    gr.HTML("""
    <div class="header-box">
        <h1 style="margin: 0; font-size: 3em; font-weight: 700;">🎯 ResNet50 ImageNet Classifier</h1>
        <p style="margin: 15px 0 0 0; font-size: 1.3em; opacity: 0.95;">
            Trained from Scratch on ImageNet-1K | 75%+ Top-1 Accuracy
        </p>
        <p style="margin: 10px 0 0 0; font-size: 1em; opacity: 0.85;">
            1000 classes • 25.6M parameters • 98MB model
        </p>
    </div>
    """)
    
    # Stats row
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="stats-card">
                <h3 style="margin: 0 0 10px 0; color: #667eea;">📊 Dataset</h3>
                <p style="margin: 5px 0;"><strong>1.28M</strong> training images</p>
                <p style="margin: 5px 0;"><strong>1000</strong> ImageNet classes</p>
            </div>
            """)
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="stats-card">
                <h3 style="margin: 0 0 10px 0; color: #667eea;">🎯 Performance</h3>
                <p style="margin: 5px 0;"><strong>75-77%</strong> top-1 accuracy</p>
                <p style="margin: 5px 0;"><strong>92-94%</strong> top-5 accuracy</p>
            </div>
            """)
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="stats-card">
                <h3 style="margin: 0 0 10px 0; color: #667eea;">⚡ Architecture</h3>
                <p style="margin: 5px 0;"><strong>ResNet50</strong> (Bottleneck)</p>
                <p style="margin: 5px 0;"><strong>25.6M</strong> parameters</p>
            </div>
            """)
    
    gr.Markdown("---")
    gr.Markdown("## 📸 Upload an Image for Classification")
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=1):
            input_images = gr.Image(
                label="Upload Image",
                type="pil",
                sources=["upload", "clipboard"],
                height=400
            )
            
            gr.Examples(
                examples=[
                    "gold_fish.png",
                    "kite.png",
                    "vulture.png",
                ],
                inputs=input_images,
                label="📌 Try these example images"
            )
            
            with gr.Row():
                topk = gr.Slider(1, 10, value=5, step=1, label="Top-K Predictions")
            
            with gr.Row():
                clear_btn = gr.Button("🔄 Clear", variant="secondary", scale=1)
                run_btn = gr.Button("🔍 Classify", variant="primary", scale=2)
        
        with gr.Column(scale=1):
            gr.HTML('<div class="prediction-box">')
            output = gr.JSON(label="🏆 Top Predictions", show_label=True)
            gr.HTML('</div>')
            
            gr.Markdown("""
            ### 💡 Tips for Best Results
            - Upload **clear, well-lit** images
            - Works best with **centered objects**
            - Supports **1000 ImageNet categories**
            - Processing time: **~1-2 seconds**
            """)
    
    # Technical accordion
    with gr.Accordion("📚 Technical Details", open=False):
        gr.Markdown("""
        ### Model Architecture
        **ResNet50** trained from scratch (no pre-trained weights) on ImageNet-1K
        
        **Training Configuration:**
        - **Optimizer:** SGD with momentum (0.9), weight decay (1e-4)
        - **Learning Rate:** Cosine annealing with warmup (0.1 → 0.0005)
        - **Augmentation:** AutoAugment (ImageNet), RandomErasing, Mixup
        - **Precision:** Mixed FP16 with gradient scaling
        - **Epochs:** 75 with early stopping
        
        **Architecture Details:**
        ```
        Input (224×224×3)
          ↓
        Conv1 (7×7, stride=2) + BN + ReLU → 112×112×64
        MaxPool (3×3, stride=2) → 56×56×64
          ↓
        Layer1: 3× Bottleneck → 56×56×256
        Layer2: 4× Bottleneck → 28×28×512
        Layer3: 6× Bottleneck → 14×14×1024
        Layer4: 3× Bottleneck → 7×7×2048
          ↓
        Global Average Pool → 1×1×2048
        Fully Connected → 1000 classes
        ```
        """)
    
    with gr.Accordion("🔗 Links & Resources", open=False):
        gr.Markdown("""
        ### Project Links
        - 🏠 [GitHub Repository](https://github.com/Malar-saravanan/tsai_assignments/tree/session9/assignment9/assignment)
        - 🤗 [Hugging Face Space](https://huggingface.co/spaces/malarsaravanan/resnet_50_1k_imagenet)
        - 🎓 [Reference Implementation](https://github.com/godsofheaven/Resnet50-from-Scratch-on-Imagenet-1K)
        - 📖 [Original ResNet Paper (He et al., 2016)](https://arxiv.org/abs/1512.03385)
        - 🗂️ [ImageNet Dataset](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
        
        ### Citation
        ```bibtex
        @inproceedings{he2016deep,
          title={Deep residual learning for image recognition},
          author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
          booktitle={CVPR},
          year={2016}
        }
        ```
        """)
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; opacity: 0.7; padding: 20px;">
        <p style="margin: 5px 0;">💜 Built with Gradio • Trained on AWS EC2 • Deployed on 🤗 Hugging Face Spaces</p>
        <p style="margin: 5px 0;">Model trained from scratch achieving 76.83% top-1 accuracy on ImageNet-1K</p>
    </div>
    """)
    
    # Button actions
    run_btn.click(fn=predict_images, inputs=[input_images, topk], outputs=output)
    clear_btn.click(lambda: (None, None), outputs=[input_images, output])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


