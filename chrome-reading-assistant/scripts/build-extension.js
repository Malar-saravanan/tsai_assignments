/**
 * Build Script for Reading Progress Saver Extension
 * 
 * This script copies and prepares the extension files for production
 */

const fs = require('fs');
const path = require('path');

// Files to copy from src to dist
const FILES_TO_COPY = [
  'manifest.json',
  'content.js',
  'background.js',
  'popup.html',
  'popup.js'
];

// Directories to copy
const DIRS_TO_COPY = [
  'icons'
];

/**
 * Copy a file from src to dist
 * @param {string} filename - Name of the file to copy
 */
function copyFile(filename) {
  const srcPath = path.join(__dirname, '..', 'src', filename);
  const distPath = path.join(__dirname, '..', 'dist', filename);
  
  if (fs.existsSync(srcPath)) {
    fs.copyFileSync(srcPath, distPath);
    console.log(`✅ Copied: ${filename}`);
  } else {
    console.warn(`⚠️  File not found: ${filename}`);
  }
}

/**
 * Copy a directory recursively
 * @param {string} dirname - Name of the directory to copy
 */
function copyDirectory(dirname) {
  const srcDir = path.join(__dirname, '..', 'src', dirname);
  const distDir = path.join(__dirname, '..', 'dist', dirname);
  
  if (fs.existsSync(srcDir)) {
    // Create dist directory if it doesn't exist
    if (!fs.existsSync(distDir)) {
      fs.mkdirSync(distDir, { recursive: true });
    }
    
    // Copy all files in the directory
    const files = fs.readdirSync(srcDir);
    files.forEach(file => {
      const srcFile = path.join(srcDir, file);
      const distFile = path.join(distDir, file);
      
      if (fs.statSync(srcFile).isDirectory()) {
        // Recursively copy subdirectories
        copyDirectory(path.join(dirname, file));
      } else {
        fs.copyFileSync(srcFile, distFile);
        console.log(`✅ Copied: ${dirname}/${file}`);
      }
    });
  } else {
    console.warn(`⚠️  Directory not found: ${dirname}`);
  }
}

/**
 * Clean the dist directory
 */
function cleanDist() {
  const distPath = path.join(__dirname, '..', 'dist');
  
  if (fs.existsSync(distPath)) {
    fs.rmSync(distPath, { recursive: true, force: true });
    console.log('🧹 Cleaned dist directory');
  }
  
  fs.mkdirSync(distPath, { recursive: true });
}

/**
 * Validate the extension files
 */
function validateExtension() {
  const distPath = path.join(__dirname, '..', 'dist');
  const manifestPath = path.join(distPath, 'manifest.json');
  
  if (!fs.existsSync(manifestPath)) {
    throw new Error('manifest.json not found in dist directory');
  }
  
  try {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    
    // Basic validation
    if (!manifest.name || !manifest.version || !manifest.manifest_version) {
      throw new Error('Invalid manifest.json: missing required fields');
    }
    
    console.log('✅ Extension validation passed');
    console.log(`📦 Extension: ${manifest.name} v${manifest.version}`);
    
  } catch (error) {
    throw new Error(`Manifest validation failed: ${error.message}`);
  }
}

/**
 * Main build function
 */
function buildExtension() {
  console.log('🔨 Building Reading Progress Saver Extension...');
  
  try {
    // Clean and recreate dist directory
    cleanDist();
    
    // Copy files
    FILES_TO_COPY.forEach(copyFile);
    
    // Copy directories
    DIRS_TO_COPY.forEach(copyDirectory);
    
    // Validate the extension
    validateExtension();
    
    console.log('🎉 Extension built successfully!');
    console.log('📁 Output directory: dist/');
    console.log('📝 To load in Chrome:');
    console.log('   1. Open chrome://extensions/');
    console.log('   2. Enable Developer mode');
    console.log('   3. Click "Load unpacked"');
    console.log('   4. Select the dist/ folder');
    
  } catch (error) {
    console.error('❌ Build failed:', error.message);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  buildExtension();
}

module.exports = { buildExtension };
