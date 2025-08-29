# 🎨 AI Image Filter Demo

A simple web application that demonstrates image convolution using custom 3x3 kernels. Upload an image and apply various filters including sharpen, blur, edge detection, emboss, or create your own custom filter.

## ✨ Features

- **Image Upload**: Drag & drop or click to upload images
- **Custom Kernels**: Create your own 3x3 convolution kernels
- **Preset Filters**: Quick access to common filters (Sharpen, Blur, Edge Detection, Emboss, Identity)
- **Real-time Processing**: Instant image filtering using HTML5 Canvas
- **Statistics**: View kernel statistics and image dimensions
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 How to Use

1. **Upload an Image**: Click the upload area or drag & drop an image file
2. **Choose a Filter**: 
   - Use preset buttons for common filters
   - Or create custom kernel by entering values in the 3x3 grid
3. **Apply Filter**: Click "Apply Filter" to process the image
4. **View Results**: See the original and filtered images side by side

## 🔧 Filter Examples

### Sharpen Filter
```
[ 0, -1,  0]
[-1,  5, -1]
[ 0, -1,  0]
```

### Blur Filter
```
[0.11, 0.11, 0.11]
[0.11, 0.11, 0.11]
[0.11, 0.11, 0.11]
```

### Edge Detection
```
[-1, -1, -1]
[-1,  8, -1]
[-1, -1, -1]
```

## 🛠️ Technical Details

- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: HTML5 Canvas API
- **Convolution Algorithm**: 3x3 kernel convolution with edge handling
- **File Support**: All common image formats (JPEG, PNG, GIF, WebP)
- **Performance**: Client-side processing, no server required

## 📱 Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## 🎯 Educational Purpose

This application demonstrates:
- Image convolution concepts
- Kernel-based image filtering
- Real-time image processing in the browser
- HTML5 Canvas manipulation
- Responsive web design

## 🚀 Running Locally

1. Clone or download the project
2. Open `index.html` in any modern web browser
3. No server setup required - runs entirely in the browser!

## 📊 Statistics Displayed

- Image dimensions and pixel count
- Kernel sum, mean, and standard deviation
- Visual kernel matrix representation

Perfect for learning about computer vision and image processing concepts!
