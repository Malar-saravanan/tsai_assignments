/**
 * Package Script for Reading Progress Saver Extension
 * 
 * This script creates a zip file ready for Chrome Web Store submission
 */

const fs = require('fs');
const path = require('path');
const archiver = require('archiver');

/**
 * Create a zip file of the extension
 * @param {string} sourceDir - Source directory to zip
 * @param {string} outputPath - Output zip file path
 */
function createZip(sourceDir, outputPath) {
  return new Promise((resolve, reject) => {
    const output = fs.createWriteStream(outputPath);
    const archive = archiver('zip', {
      zlib: { level: 9 } // Maximum compression
    });

    output.on('close', () => {
      const size = (archive.pointer() / 1024 / 1024).toFixed(2);
      console.log(`📦 Extension packaged: ${outputPath} (${size} MB)`);
      resolve();
    });

    archive.on('error', (err) => {
      reject(err);
    });

    archive.pipe(output);

    // Add all files from dist directory
    archive.directory(sourceDir, false);

    archive.finalize();
  });
}

/**
 * Main packaging function
 */
async function packageExtension() {
  console.log('📦 Packaging Reading Progress Saver Extension...');
  
  try {
    const distPath = path.join(__dirname, '..', 'dist');
    const packagePath = path.join(__dirname, '..', 'reading-progress-saver.zip');
    
    // Check if dist directory exists
    if (!fs.existsSync(distPath)) {
      throw new Error('dist/ directory not found. Run "npm run build" first.');
    }
    
    // Create zip file
    await createZip(distPath, packagePath);
    
    console.log('🎉 Extension packaged successfully!');
    console.log('📁 Package file: reading-progress-saver.zip');
    console.log('📝 Ready for Chrome Web Store submission');
    
  } catch (error) {
    console.error('❌ Packaging failed:', error.message);
    
    if (error.message.includes('archiver')) {
      console.log('💡 Install archiver: npm install archiver');
    }
    
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  packageExtension();
}

module.exports = { packageExtension };
