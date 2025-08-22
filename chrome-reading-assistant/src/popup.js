// Reading Progress Saver - Popup Script

document.addEventListener('DOMContentLoaded', function () {
  // Initialize popup
  loadCurrentProgress();
  loadSavedPages();
  loadTrackingStatus();
  setupEventListeners();
});

function setupEventListeners () {
  // Toggle tracking for current page
  document.getElementById('toggleTracking').addEventListener('click', async () => {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (tab) {
        await chrome.tabs.sendMessage(tab.id, { action: 'toggleTracking' });
        // Reload status after a short delay
        setTimeout(() => {
          loadTrackingStatus();
        }, 500);
      }
    } catch (error) {
      showStatus('Error toggling tracking', 'error');
    }
  });

  // Clear current page progress
  document.getElementById('clearCurrent').addEventListener('click', async () => {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (tab) {
        await chrome.tabs.sendMessage(tab.id, { action: 'clearProgress' });
        showStatus('Current page progress cleared!', 'success');
        loadCurrentProgress();
      }
    } catch (error) {
      showStatus('Error clearing current progress', 'error');
    }
  });

  // Clear all progress
  document.getElementById('clearAll').addEventListener('click', async () => {
    if (confirm('Are you sure you want to clear all saved reading progress? This cannot be undone.')) {
      try {
        const response = await chrome.runtime.sendMessage({ action: 'clearAllProgress' });

        if (response.success) {
          showStatus(`Cleared ${response.count} saved pages!`, 'success');
          loadSavedPages();
        } else {
          showStatus('Error clearing all progress', 'error');
        }
      } catch (error) {
        showStatus('Error clearing all progress', 'error');
      }
    }
  });
}

async function loadCurrentProgress () {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (tab) {
      const response = await chrome.tabs.sendMessage(tab.id, { action: 'getCurrentProgress' });

      if (response) {
        updateProgressDisplay(response);
      } else {
        showNoProgress();
      }
    } else {
      showNoProgress();
    }
  } catch (error) {
    showNoProgress();
  }
}

function updateProgressDisplay (data) {
  const progressFill = document.getElementById('progressFill');
  const progressText = document.getElementById('progressText');

  const progress = Math.max(0, Math.min(100, data.progress));

  progressFill.style.width = `${progress}%`;
  progressText.textContent = `${progress}% read (${data.scrollPosition}px / ${data.documentHeight}px)`;
}

function showNoProgress () {
  const progressFill = document.getElementById('progressFill');
  const progressText = document.getElementById('progressText');

  progressFill.style.width = '0%';
  progressText.textContent = 'No progress saved for this page';
}

async function loadSavedPages () {
  try {
    const savedPages = await chrome.runtime.sendMessage({ action: 'getAllProgress' });
    displaySavedPages(savedPages);
  } catch (error) {
    displaySavedPages([]);
  }
}

function displaySavedPages (pages) {
  const pageList = document.getElementById('pageList');

  if (pages.length === 0) {
    pageList.innerHTML = `
      <div class="empty-state">
        <p>📚 No saved reading progress yet</p>
        <p>Start reading on any webpage and your progress will be automatically saved!</p>
      </div>
    `;
    return;
  }

  const pagesHtml = pages.slice(0, 10).map(page => {
    const timeAgo = getTimeAgo(page.timestamp);
    const domain = new URL(page.url).hostname;
    const statusIcon = page.isEnabled ? '🟢' : '🔴';

    return `
      <div class="page-item" data-url="${page.url}">
        <div class="page-title">${page.title || 'Untitled Page'}</div>
        <div class="page-url">${domain} ${statusIcon}</div>
        <div class="page-time">${timeAgo} • ${Math.round(page.scrollPosition)}px</div>
      </div>
    `;
  }).join('');

  pageList.innerHTML = pagesHtml;

  // Add click listeners to page items
  pageList.querySelectorAll('.page-item').forEach(item => {
    item.addEventListener('click', () => {
      const url = item.dataset.url;
      chrome.tabs.create({ url });
      window.close();
    });
  });
}

function getTimeAgo (timestamp) {
  const now = Date.now();
  const diff = now - timestamp;

  const minutes = Math.floor(diff / (1000 * 60));
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));

  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;

  return new Date(timestamp).toLocaleDateString();
}

function showStatus (message, type = 'info') {
  const status = document.getElementById('status');
  status.textContent = message;
  status.style.color = type === 'success' ? '#4CAF50' : type === 'error' ? '#f44336' : 'rgba(255, 255, 255, 0.8)';

  // Clear status after 3 seconds
  setTimeout(() => {
    status.textContent = '';
  }, 3000);
}

// Load tracking status for current page
async function loadTrackingStatus () {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (tab) {
      const result = await chrome.storage.local.get('enabledPages');
      const enabledPages = result.enabledPages || {};
      const isEnabled = enabledPages[tab.url] || false;

      const statusIndicator = document.getElementById('statusIndicator');
      const statusText = document.getElementById('statusText');

      if (isEnabled) {
        statusIndicator.className = 'status-indicator status-enabled';
        statusText.textContent = 'Tracking enabled for this page';
      } else {
        statusIndicator.className = 'status-indicator status-disabled';
        statusText.textContent = 'Tracking disabled for this page';
      }
    }
  } catch (error) {
    console.error('Error loading tracking status:', error);
  }
}

// Refresh data when popup is focused
document.addEventListener('focus', () => {
  loadCurrentProgress();
  loadSavedPages();
  loadTrackingStatus();
});
