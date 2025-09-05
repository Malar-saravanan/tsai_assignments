# 🐾 WildSnap

A modern FastAPI web application for discovering amazing animals with real photos and managing file uploads with detailed analysis.

## ✨ Features

- **Animal Discovery**: Explore Cat 🐱, Dog 🐶, and Elephant 🐘 with stunning real photos
- **Smart File Upload**: Drag & drop files with instant detailed analysis
- **Interactive UI**: Beautiful clickable animal cards with smooth animations
- **Real Photography**: High-quality animal images instead of simple emojis
- **Secure Handling**: File validation and safe upload processing
- **Responsive Design**: Perfect on desktop and mobile devices

## 🛠️ Tech Stack

- **FastAPI** - High-performance Python web framework
- **Uvicorn** - Lightning-fast ASGI server
- **HTML5/CSS3/JavaScript** - Modern responsive frontend
- **Jinja2** - Powerful templating engine
- **aiofiles** - Async file operations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation & Run

#### Option 1: Virtual Environment (Recommended)

```powershell
# Windows PowerShell
# Clone the repository
git clone <your-repo-url>
cd animal-app

# Create virtual environment
python -m venv venv

# Install dependencies directly
.\venv\Scripts\python.exe -m 

is no
# Start WildSnap using uvicorn
.\venv\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 8001 --reload
```

```bash
# macOS/Linux
# Clone the repository
git clone <your-repo-url>
cd animal-app

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn[standard] python-multipart jinja2 aiofiles

# Start WildSnap using uvicorn
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

#### Option 2: Direct pip installation

```bash
# Clone the repository
git clone <your-repo-url>
cd animal-app

# Install dependencies
pip install -r requirements.txt

# Start WildSnap using uvicorn
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

**🌐 Open in browser:**
- **App**: `http://localhost:8001`
- **API Docs**: `http://localhost:8001/docs`

## 📁 Project Structure

```
animal-app/
├── main.py           # FastAPI application
├── pyproject.toml    # Project configuration
├── templates/
│   └── index.html   # Frontend template
├── static/images/   # Animal photos
└── uploads/         # User file uploads
```

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main application |
| `GET` | `/api/animal/{type}` | Animal info (cat/dog/elephant) |
| `POST` | `/api/upload` | File upload & analysis |
| `GET` | `/docs` | Interactive API documentation |

## 🎯 How to Use

1. **🐾 Discover Animals**: Click any animal card to see real photos and fascinating facts
2. **📁 Upload Files**: Drag & drop or click to upload - get instant file details
3. **📊 View Analysis**: See file name, size, type, and format information

## 🚀 Production Deployment

### Using Virtual Environment

```powershell
# Windows PowerShell - Basic production server
.\venv\Scripts\uvicorn.exe main:app --host 0.0.0.0 --port 8000

# Windows PowerShell - With multiple workers
.\venv\Scripts\python.exe -m pip install gunicorn
.\venv\Scripts\gunicorn.exe main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

```bash
# macOS/Linux - Basic production server
uvicorn main:app --host 0.0.0.0 --port 8000

# macOS/Linux - With multiple workers  
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Direct Installation

```bash
# Basic production server
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# With multiple workers
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
