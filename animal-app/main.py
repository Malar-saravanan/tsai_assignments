from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import mimetypes
from pathlib import Path
import aiofiles
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="WildSnap", description="Discover amazing animals and manage your files with style")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff', 'tif', 'svg', 'doc', 'docx', 'mp3', 'mp4', 'avi', 'mov', 'zip', 'rar', 'csv', 'xlsx'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size_string(size_bytes: int) -> str:
    """Convert file size to human readable format"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/animal/{animal_type}")
async def get_animal_info(animal_type: str):
    """Return animal information based on type"""
    valid_animals = ['cat', 'dog', 'elephant']
    if animal_type.lower() not in valid_animals:
        raise HTTPException(status_code=400, detail="Invalid animal type")
    
    # Animal data with emoji and description
    animal_data = {
        'cat': {
            'emoji': '🐱',
            'name': 'Cat',
            'description': 'Cats are curious, independent, and make wonderful companions!',
            'facts': ['Cats sleep 12-16 hours a day', 'They have excellent night vision', 'Cats can rotate their ears 180 degrees']
        },
        'dog': {
            'emoji': '🐶',
            'name': 'Dog', 
            'description': 'Dogs are loyal, friendly, and known as man\'s best friend!',
            'facts': ['Dogs have an incredible sense of smell', 'They can learn over 150 words', 'Dogs are pack animals by nature']
        },
        'elephant': {
            'emoji': '🐘',
            'name': 'Elephant',
            'description': 'Elephants are intelligent, gentle giants with amazing memories!',
            'facts': ['Elephants can weigh up to 6 tons', 'They have excellent memories', 'Elephants are highly social animals']
        }
    }
    
    animal_info = animal_data[animal_type.lower()]
    return {
        'animal': animal_type.lower(),
        'emoji': animal_info['emoji'],
        'name': animal_info['name'],
        'description': animal_info['description'],
        'facts': animal_info['facts'],
        'image_url': f'/static/images/{animal_type.lower()}.jpg',
        'message': f'Here is information about the amazing {animal_info["name"]}!'
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and return file information"""
    logger.info(f"Upload request received for file: {file.filename}")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    # Check if file extension is allowed
    if not allowed_file(file.filename):
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'no extension'
        logger.warning(f"File type not allowed: {file_ext}")
        raise HTTPException(
            status_code=400, 
            detail=f'File type "{file_ext}" not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        )
    
    # Save the file
    file_path = UPLOAD_FOLDER / file.filename
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    logger.info(f"File saved to: {file_path}")
    
    # Get file information
    file_size = file_path.stat().st_size
    file_type = mimetypes.guess_type(str(file_path))[0] or 'unknown'
    
    result = {
        'success': True,
        'filename': file.filename,
        'file_size': get_file_size_string(file_size),
        'file_size_bytes': file_size,
        'file_type': file_type,
        'message': f'File "{file.filename}" uploaded successfully!'
    }
    
    logger.info(f"File upload successful: {file.filename}")
    return result

@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """Serve uploaded files"""
    file_path = UPLOAD_FOLDER / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    print("🐾 Starting WildSnap on http://localhost:8001")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
