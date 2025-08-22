/**
 * Reading Progress Saver - Content Script
 *
 * This script runs on every webpage and handles:
 * - Reading progress tracking
 * - Progress restoration
 * - Toggle button UI
 * - Communication with background script
 *
 * @version 1.0.0
 * @author Your Name
 * @license MIT
 */

/**
 * Main class for tracking reading progress on web pages
 */
class ReadingProgressTracker {
  /**
   * Initialize the reading progress tracker
   * @param {Object} options - Configuration options
   */
  constructor (options = {}) {
    this.currentUrl = window.location.href;
    this.scrollPosition = 0;
    this.timestamp = Date.now();
    this.isRestoring = false;
    this.isEnabled = false;
    this.options = {
      saveDelay: 1000, // Debounce delay for saving
      restoreDelay: 500, // Delay before restoring progress
      notificationDuration: 3000, // How long notifications show
      ...options
    };

    this.init();
  }

  /**
   * Initialize the tracker
   * @private
   */
  async init () {
    try {
      await this.checkIfEnabled();
      await this.checkSavedProgress();
      this.setupScrollTracking();
      this.setupVisibilityTracking();
      this.setupBeforeUnload();
      this.addToggleButton();
    } catch (error) {
      // ...existing code...
    }
  }

  /**
   * Check if tracking is enabled for this page
   * @private
   */
  async checkIfEnabled () {
    try {
      const result = await chrome.storage.local.get('enabledPages');
      const enabledPages = result.enabledPages || {};
      this.isEnabled = enabledPages[this.currentUrl] || false;
    } catch (error) {
      // ...existing code...
      this.isEnabled = false;
    }
  }

  /**
   * Check and restore saved progress for this page
   * @private
   */
  async checkSavedProgress () {
    try {
      if (!this.isEnabled) return;

      const result = await chrome.storage.local.get(this.currentUrl);
      const savedData = result[this.currentUrl];

      if (savedData && savedData.scrollPosition) {
        setTimeout(() => {
          this.restoreProgress(savedData);
        }, this.options.restoreDelay);
      }
    } catch (error) {
      // ...existing code...
    }
  }

  /**
   * Restore reading progress to saved position
   * @param {Object} savedData - Saved progress data
   * @private
   */
  restoreProgress (savedData) {
    this.isRestoring = true;

    window.scrollTo({
      top: savedData.scrollPosition,
      behavior: 'smooth'
    });

    this.showRestoreIndicator(savedData.scrollPosition);

    setTimeout(() => {
      this.isRestoring = false;
    }, 1000);
  }

  /**
   * Show notification when progress is restored
   * @param {number} scrollPosition - The restored scroll position
   * @private
   */
  showRestoreIndicator (scrollPosition) {
    const indicator = document.createElement('div');
    indicator.id = 'reading-progress-restore-notification';
    indicator.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: #4CAF50;
      color: white;
      padding: 12px 16px;
      border-radius: 8px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 14px;
      font-weight: 500;
      z-index: 2147483647;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      animation: reading-progress-slideIn 0.5s ease-out;
      max-width: 300px;
    `;

    indicator.innerHTML = `
      <div style="display: flex; align-items: center; gap: 8px;">
        <span>📖</span>
        <div>
          <div>Reading progress restored!</div>
          <div style="font-size: 12px; opacity: 0.9; margin-top: 2px;">
            Position: ${Math.round(scrollPosition)}px
          </div>
        </div>
      </div>
    `;

    // Add CSS animation
    if (!document.getElementById('reading-progress-styles')) {
      const style = document.createElement('style');
      style.id = 'reading-progress-styles';
      style.textContent = `
        @keyframes reading-progress-slideIn {
          from { transform: translateX(100%); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
        @keyframes reading-progress-slideOut {
          from { transform: translateX(0); opacity: 1; }
          to { transform: translateX(100%); opacity: 0; }
        }
      `;
      document.head.appendChild(style);
    }

    document.body.appendChild(indicator);

    // Remove after duration
    setTimeout(() => {
      if (indicator.parentNode) {
        indicator.style.animation = 'reading-progress-slideOut 0.5s ease-in';
        setTimeout(() => {
          if (indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
          }
        }, 500);
      }
    }, this.options.notificationDuration);
  }

  /**
   * Set up scroll tracking with debouncing
   * @private
   */
  setupScrollTracking () {
    let scrollTimeout;

    const handleScroll = () => {
      if (this.isRestoring) return;

      this.scrollPosition = window.pageYOffset;
      this.timestamp = Date.now();

      if (this.isEnabled) {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(() => {
          this.saveProgress();
        }, this.options.saveDelay);
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
  }

  /**
   * Set up visibility change tracking
   * @private
   */
  setupVisibilityTracking () {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'hidden' && this.isEnabled) {
        this.saveProgress();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
  }

  /**
   * Set up beforeunload event
   * @private
   */
  setupBeforeUnload () {
    const handleBeforeUnload = () => {
      if (this.isEnabled) {
        this.saveProgress();
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
  }

  /**
   * Save current reading progress
   * @private
   */
  async saveProgress () {
    try {
      const progressData = {
        scrollPosition: this.scrollPosition,
        timestamp: this.timestamp,
        title: document.title,
        url: this.currentUrl
      };

      await chrome.storage.local.set({
        [this.currentUrl]: progressData
      });

      // ...existing code...
    } catch (error) {
      // ...existing code...
    }
  }

  /**
   * Clear progress for current page
   * @public
   */
  async clearProgress () {
    try {
      await chrome.storage.local.remove(this.currentUrl);
      // ...existing code...
    } catch (error) {
      // ...existing code...
    }
  }

  /**
   * Add toggle button to the page
   * @private
   */
  addToggleButton () {
    // Remove any existing button
    const oldBtn = document.getElementById('reading-progress-toggle');
    if (oldBtn) oldBtn.remove();

    const button = document.createElement('div');
    button.id = 'reading-progress-toggle';
    button.style.cssText = `
      position: fixed;
      top: 20px;
      left: 20px;
      background: ${this.isEnabled ? '#4CAF50' : '#f44336'};
      color: white;
      padding: 12px 16px;
      border-radius: 25px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      z-index: 2147483646;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      transition: all 0.3s ease;
      user-select: none;
      border: none;
      outline: none;
      display: flex;
      align-items: center;
      gap: 8px;
    `;

    // Create ON/OFF text span
    const statusSpan = document.createElement('span');
    statusSpan.id = 'reading-progress-status';
    statusSpan.textContent = this.isEnabled ? '📖 ON' : '📖 OFF';
    button.appendChild(statusSpan);

    button.addEventListener('click', () => {
      this.toggleTracking();
    });

    button.addEventListener('mouseenter', () => {
      button.style.transform = 'scale(1.05)';
    });

    button.addEventListener('mouseleave', () => {
      button.style.transform = 'scale(1)';
    });

    document.body.appendChild(button);
  }

  /**
   * Toggle tracking for this page
   * @public
   */
  async toggleTracking () {
    try {
      const result = await chrome.storage.local.get('enabledPages');
      const enabledPages = result.enabledPages || {};

      this.isEnabled = !this.isEnabled;
      enabledPages[this.currentUrl] = this.isEnabled;

      await chrome.storage.local.set({ enabledPages });

      // Update button appearance and status
      this.addToggleButton();
      this.showToggleNotification();

      // ...existing code...
    } catch (error) {
      // ...existing code...
    }
  }

  /**
   * Show toggle notification
   * @private
   */
  showToggleNotification () {
    const notification = document.createElement('div');
    notification.id = 'reading-progress-toggle-notification';
    notification.style.cssText = `
      position: fixed;
      top: 80px;
      left: 20px;
      background: ${this.isEnabled ? '#4CAF50' : '#f44336'};
      color: white;
      padding: 12px 16px;
      border-radius: 8px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 14px;
      font-weight: 500;
      z-index: 2147483647;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      animation: reading-progress-slideIn 0.5s ease-out;
    `;

    notification.innerHTML = `📖 Reading progress tracking ${this.isEnabled ? 'enabled' : 'disabled'}`;

    document.body.appendChild(notification);

    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, this.options.notificationDuration);
  }
}

// Initialize the tracker when the page loads
let readingProgressTracker;

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    readingProgressTracker = new ReadingProgressTracker();
    window.readingProgressTracker = readingProgressTracker;
  });
} else {
  readingProgressTracker = new ReadingProgressTracker();
  window.readingProgressTracker = readingProgressTracker;
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  try {
    switch (request.action) {
    case 'getCurrentProgress':
      sendResponse({
        scrollPosition: window.pageYOffset,
        documentHeight: document.documentElement.scrollHeight,
        viewportHeight: window.innerHeight,
        progress: Math.round((window.pageYOffset / (document.documentElement.scrollHeight - window.innerHeight)) * 100)
      });
      break;

    case 'clearProgress':
      chrome.storage.local.remove(window.location.href, () => {
        // Optionally, reload the page or update UI
        if (window.readingProgressTracker) {
          window.readingProgressTracker.scrollPosition = 0;
        }
        sendResponse({ success: true });
      });
      return true;

    case 'toggleTracking':
      if (window.readingProgressTracker) {
        window.readingProgressTracker.toggleTracking();
      }
      sendResponse({ success: true });
      break;

    default:
      sendResponse({ error: 'Unknown action' });
    }
  } catch (error) {
  // ...existing code...
    sendResponse({ error: error.message });
  }
});
