# 🚀 Upload to Hugging Face Spaces

## 📦 Files to Upload

Upload these files from the `hf_app/` folder:

### Required Files (Upload ALL of these):

1. ✅ **app.py** (9.9KB) - Main Gradio interface
2. ✅ **model_quantized.pt** (330MB) - Quantized model ⭐ USE THIS
3. ✅ **requirements.txt** - Dependencies
4. ✅ **README.md** - Space description
5. ✅ **.gitattributes** - Git LFS config

### ❌ DO NOT Upload:
- ❌ `final_step_1700_loss_0.094349.pt` (1.4GB) - Too large, use quantized instead
- ❌ `quantize_model.py` - Only needed locally

---

## 🎯 Step-by-Step Upload Guide

### Step 1: Create Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   - **Space name**: `shakespeare-gpt2-generator`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free)
   - **Visibility**: Public

### Step 2: Clone Your Space Repository

```bash
# Clone the empty space
git clone https://huggingface.co/spaces/YOUR_USERNAME/shakespeare-gpt2-generator
cd shakespeare-gpt2-generator
```

### Step 3: Copy Files

```bash
# Copy all required files from hf_app/
cp /Users/msaravanan/Desktop/assignment/hf_app/app.py .
cp /Users/msaravanan/Desktop/assignment/hf_app/model_quantized.pt .
cp /Users/msaravanan/Desktop/assignment/hf_app/requirements.txt .
cp /Users/msaravanan/Desktop/assignment/hf_app/README.md .
cp /Users/msaravanan/Desktop/assignment/hf_app/.gitattributes .
```

### Step 4: Setup Git LFS

```bash
# Install Git LFS if not already installed
brew install git-lfs  # Mac
# OR download from: https://git-lfs.github.com/

# Initialize LFS
git lfs install

# Track the model file
git lfs track "*.pt"

# Verify LFS is tracking
git lfs ls-files
```

### Step 5: Push to Hugging Face

```bash
# Add all files
git add .

# Commit
git commit -m "Initial commit: Shakespeare GPT-2 (124M, loss 0.094349, FP16 quantized)"

# Push to Hugging Face
git push origin main
```

**⏱️ Upload time**: ~2-5 minutes (330MB model)

### Step 6: Wait for Build

1. Go to: `https://huggingface.co/spaces/YOUR_USERNAME/shakespeare-gpt2-generator`
2. Wait 2-3 minutes for app to build
3. App will automatically start!

---

## ✅ Verification Checklist

After upload, verify:

- [ ] Space URL is accessible
- [ ] Model loads successfully (check logs)
- [ ] Gradio interface appears with teal theme
- [ ] "Generate Shakespeare" button works
- [ ] Example prompts generate text
- [ ] Model details show at bottom

---

## 🎨 Your Live Space

Once deployed, your app will be at:

```
https://huggingface.co/spaces/YOUR_USERNAME/shakespeare-gpt2-generator
```

Share this link to showcase your model! 🎉

---

## 🐛 Troubleshooting

### Issue: "Model not found"
**Solution**: Ensure `model_quantized.pt` is in the same directory as `app.py`

### Issue: "Out of memory"
**Solution**: The 330MB quantized model runs fine on free CPU tier!

### Issue: "Git LFS bandwidth limit"
**Solution**: HF provides unlimited LFS for model files - you're good! ✅

### Issue: "Build failed"
**Solution**: Check the "Logs" tab in your Space for error details

---

## 📊 What Gets Deployed

- **Model**: 330MB quantized (FP16)
- **Loss**: 0.094349 (meets target ✅)
- **Interface**: Teal-themed Gradio
- **Cost**: FREE (CPU Basic tier)

---

## 🎯 Quick Commands Summary

```bash
# One-time setup
git clone https://huggingface.co/spaces/YOUR_USERNAME/shakespeare-gpt2-generator
cd shakespeare-gpt2-generator
git lfs install
git lfs track "*.pt"

# Copy files
cp /Users/msaravanan/Desktop/assignment/hf_app/{app.py,model_quantized.pt,requirements.txt,README.md,.gitattributes} .

# Upload
git add .
git commit -m "Initial commit: Shakespeare GPT-2"
git push origin main
```

---

## 🔗 Next Steps

After deployment:
1. ✅ Test the live app
2. 📸 Take screenshots for your GitHub
3. 🔗 Share the Space link
4. 📊 Monitor usage in Analytics tab

Good luck! 🚀

