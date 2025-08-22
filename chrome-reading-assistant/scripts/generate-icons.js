/**
 * Icon Generator for Reading Progress Saver Extension
 * 
 * This script generates the required icon sizes for the Chrome extension
 * using Canvas API to create programmatic icons.
 */

const fs = require('fs');
const path = require('path');

// Icon sizes required by Chrome extension
const ICON_SIZES = [16, 32, 48, 128];

// Create a simple HTML file to generate icons using Canvas
function createIconGeneratorHTML() {
  const html = `
<!DOCTYPE html>
<html>
<head>
    <title>Icon Generator</title>
    <style>
        body { margin: 20px; font-family: Arial, sans-serif; }
        .canvas-container { margin: 20px 0; }
        canvas { border: 1px solid #ccc; margin: 10px; }
    </style>
</head>
<body>
    <h1>Reading Progress Saver - Icon Generator</h1>
    <p>Generating icons for sizes: ${ICON_SIZES.join(', ')}px</p>
    
    ${ICON_SIZES.map(size => `
    <div class="canvas-container">
        <h3>${size}x${size}px</h3>
        <canvas id="canvas${size}" width="${size}" height="${size}"></canvas>
    </div>
    `).join('')}
    
    <script>
        function drawIcon(canvas, size) {
            const ctx = canvas.getContext('2d');
            
            // Background gradient
            const gradient = ctx.createLinearGradient(0, 0, size, size);
            gradient.addColorStop(0, '#667eea');
            gradient.addColorStop(1, '#764ba2');
            
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, size, size);
            
            // Book icon (simplified)
            ctx.fillStyle = 'white';
            ctx.font = \`\${size * 0.5}px Arial\`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('📖', size/2, size/2);
            
            // Progress indicator bar
            ctx.fillStyle = '#4CAF50';
            ctx.fillRect(size * 0.1, size * 0.8, size * 0.8, size * 0.1);
        }
        
        function downloadCanvas(canvas, filename) {
            const link = document.createElement('a');
            link.download = filename;
            link.href = canvas.toDataURL('image/png');
            link.click();
        }
        
        // Draw all icons
        ${ICON_SIZES.map(size => `
        const canvas${size} = document.getElementById('canvas${size}');
        drawIcon(canvas${size}, ${size});
        downloadCanvas(canvas${size}, 'icon${size}.png');
        `).join('')}
        
        console.log('Icons generated successfully!');
    </script>
</body>
</html>`;

  return html;
}

// Main function
function generateIcons() {
  console.log('🎨 Generating icons for Reading Progress Saver...');
  
  // Create icons directory if it doesn't exist
  const iconsDir = path.join(__dirname, '..', 'src', 'icons');
  if (!fs.existsSync(iconsDir)) {
    fs.mkdirSync(iconsDir, { recursive: true });
  }
  
  // Create the HTML generator
  const generatorHTML = createIconGeneratorHTML();
  const generatorPath = path.join(__dirname, 'icon-generator.html');
  
  fs.writeFileSync(generatorPath, generatorHTML);
  
  console.log('✅ Icon generator HTML created at:', generatorPath);
  console.log('📝 To generate icons:');
  console.log('   1. Open icon-generator.html in a browser');
  console.log('   2. Icons will be automatically downloaded');
  console.log('   3. Move the downloaded icons to src/icons/');
  console.log('   4. Delete icon-generator.html when done');
  
  // Create placeholder icon files
  ICON_SIZES.forEach(size => {
    const iconPath = path.join(iconsDir, `icon${size}.png`);
    if (!fs.existsSync(iconPath)) {
      // Create a simple text file as placeholder
      fs.writeFileSync(iconPath, `# Placeholder for icon${size}.png\n# Replace with actual ${size}x${size} PNG icon`);
      console.log(`📄 Created placeholder: icon${size}.png`);
    }
  });
  
  console.log('🎉 Icon generation setup complete!');
}

// Run if called directly
if (require.main === module) {
  generateIcons();
}

module.exports = { generateIcons, ICON_SIZES };
