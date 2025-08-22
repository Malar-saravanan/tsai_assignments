
# 📖 Reading Progress Saver - Chrome Extension

A Chrome extension that saves and restores your reading progress on any website. Never lose your place again!

[![Chrome Web Store](https://img.shields.io/badge/Chrome%20Web%20Store-Reading%20Progress%20Saver-blue?logo=google-chrome)](https://chrome.google.com/webstore/detail/reading-progress-saver/your-extension-id)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/yourusername/reading-progress-saver)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Chrome Version](https://img.shields.io/badge/Chrome-88+-green?logo=google-chrome)](https://www.google.com/chrome/)

## ✨ Features

- **Selective Tracking**: Enable/disable tracking for any page with a floating ON/OFF button
- **Automatic Progress Tracking**: Saves your scroll position as you read
- **Smart Restoration**: Automatically restores your reading position when you revisit a page
- **Visual Notifications**: Subtle popup when progress is restored
- **Progress Management**: Popup interface to view and clear saved progress
- **Privacy-First**: All data stored locally in your browser
- **Performance Optimized**: Lightweight and fast

## 🚀 Quick Start

### Installation

#### Chrome Web Store (Recommended)
1. Visit the [Chrome Web Store](https://chrome.google.com/webstore/detail/reading-progress-saver/your-extension-id)
2. Click "Add to Chrome"
3. Confirm the installation

#### Developer Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/reading-progress-saver.git
   cd reading-progress-saver
   ```
2. **Install dependencies**
   ```bash
   npm install
   ```
3. **Build the extension**
   ```bash
   npm run build
   ```
4. **Load in Chrome**
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select the `dist/` folder

## 🎯 How to Use

1. **Enable Tracking**: Click the floating ON/OFF button on any page you want to track
2. **Read**: Your scroll position is saved automatically
3. **Return Later**: Your position is restored automatically
4. **Manage Progress**: Use the extension popup to view or clear saved progress

## 🛠️ For Developers

- Code is modular and documented for easy extension
- To add features, edit files in `src/` and use the build scripts
- Run `npm run lint` to check code quality
- Pull requests and contributions are welcome!

## 📄 License

MIT — free for personal and commercial use

### Advanced Features
- **Toggle Anytime**: Use the popup or floating button to enable/disable tracking for any page
- **Progress Overview**: View all your saved pages with timestamps and progress indicators
- **Quick Navigation**: Click on saved pages in the popup to quickly return to them
- **Bulk Management**: Clear individual page progress or all progress at once

## 📱 Features in Detail

### Selective Progress Tracking
- Only tracks pages you explicitly enable
- Floating toggle button on every page for easy control
- Saves progress when you leave the page or close the tab
- Debounced saving to avoid performance issues

### Smart Restoration
- Automatically scrolls to your saved position when returning to a page
- Smooth scrolling animation for better user experience
- Visual notification showing restoration with position details

### Progress Management
- View current page progress in the popup
- See all recently saved pages with timestamps
- Clear individual page progress or all progress at once
- Click on saved pages to quickly navigate back

### Privacy & Performance
- All data stored locally in Chrome's storage
- No data sent to external servers
- Automatic cleanup of old entries to manage storage
- Lightweight and fast

## 🛠️ Development

### Prerequisites
- Node.js 16.0.0 or higher
- Chrome browser 88 or higher

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/reading-progress-saver.git
cd reading-progress-saver

# Install dependencies
npm install

# Build the extension
npm run build

# Load in Chrome (see installation instructions above)
```

### Available Scripts
```bash
npm run build          # Build the extension for production
npm run dev            # Build and show loading instructions
npm run lint           # Run ESLint to check code quality
npm run lint:fix       # Fix ESLint issues automatically
npm run package        # Create a zip file for Chrome Web Store
```

### Project Structure
```
reading-progress-saver/
├── src/                    # Source files
│   ├── manifest.json      # Extension configuration
│   ├── content.js         # Content script for page interaction
│   ├── background.js      # Background service worker
│   ├── popup.html        # Extension popup interface
│   ├── popup.js          # Popup functionality
│   └── icons/            # Extension icons
├── dist/                  # Built extension (generated)
├── scripts/              # Build and utility scripts
├── package.json          # Project configuration
└── README.md            # This file
```

## 🔧 Technical Details

### Architecture
- **Manifest V3**: Uses the latest Chrome extension manifest version
- **Service Worker**: Background script for data management
- **Content Scripts**: Page interaction and progress tracking
- **Popup Interface**: User-friendly management interface

### Permissions Used
- `storage`: To save reading progress locally
- `activeTab`: To interact with the current tab
- `<all_urls>`: To work on all websites

### Data Storage
- **Local Storage**: All data stored in Chrome's local storage
- **Selective Tracking**: Only enabled pages are tracked
- **Automatic Cleanup**: Old entries are automatically removed

### Performance Optimizations
- **Debounced Saving**: Prevents excessive storage writes
- **Passive Event Listeners**: Optimized scroll event handling
- **Efficient DOM Queries**: Minimal DOM manipulation
- **Memory Management**: Proper cleanup of event listeners

## 🎨 Customization

The extension is designed to be easily customizable:

### Colors and Styling
- Modify CSS in `src/popup.html` to change the color scheme
- Update button styles in `src/content.js`
- Customize notification appearance

### Behavior
- Adjust saving frequency in `src/content.js` (options.saveDelay)
- Modify notification duration (options.notificationDuration)
- Change restoration delay (options.restoreDelay)

### UI
- Customize the popup interface in `src/popup.html` and `src/popup.js`
- Modify toggle button appearance
- Update notification messages

## 🐛 Troubleshooting

### Extension Not Working
1. Make sure the extension is enabled in `chrome://extensions/`
2. Check the browser console for any error messages
3. Try refreshing the page you're reading
4. Verify the extension has the necessary permissions

### Progress Not Saving
1. Ensure the extension has permission to access the website
2. Check if the page has enough content to scroll
3. Verify the extension is pinned and active
4. Make sure tracking is enabled for the page (check the toggle button)

### Performance Issues
1. The extension automatically limits saved pages to prevent storage bloat
2. If you experience slowdowns, try clearing some saved progress
3. Check if you have too many enabled pages

### Common Issues
- **Button not appearing**: Refresh the page or check if the extension is loaded
- **Progress not restoring**: Ensure tracking was enabled when you left the page
- **Popup not working**: Try reloading the extension

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Reporting Issues
- Use the [GitHub Issues](https://github.com/yourusername/reading-progress-saver/issues) page
- Include detailed steps to reproduce the problem
- Provide your Chrome version and operating system

### Suggesting Features
- Open a new issue with the "enhancement" label
- Describe the feature and its benefits
- Consider implementation complexity

### Code Contributions
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`npm run lint`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines
- Follow the existing code style
- Add JSDoc comments for new functions
- Test your changes thoroughly
- Update documentation if needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Chrome Extensions API documentation
- The open-source community for inspiration and tools
- All contributors and users who provide feedback

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/reading-progress-saver/issues)
- **Email**: your.email@example.com
- **Documentation**: [Wiki](https://github.com/yourusername/reading-progress-saver/wiki)

## 🗺️ Roadmap

### Version 1.1.0 (Planned)
- [ ] Export/import progress data
- [ ] Keyboard shortcuts
- [ ] Reading time estimation
- [ ] Dark mode support

### Version 1.2.0 (Future)
- [ ] Sync across devices
- [ ] Reading statistics
- [ ] Custom themes
- [ ] Advanced filtering

---

**Made with ❤️ for readers everywhere**

If you find this extension helpful, please consider:
- ⭐ Starring this repository
- 📝 Leaving a review on the Chrome Web Store
- 💝 Supporting the project

Happy reading! 📚
