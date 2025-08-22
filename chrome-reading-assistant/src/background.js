// Reading Progress Saver - Background Service Worker

// Handle extension installation
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
  // ...existing code...

    // Set default settings
    chrome.storage.local.set({
      settings: {
        autoSave: true,
        showNotifications: true,
        maxSavedPages: 100
      }
    });
  }
});

// Handle messages from content scripts and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getAllProgress') {
    getAllProgress().then(sendResponse);
    return true; // Keep message channel open for async response
  } else if (request.action === 'clearAllProgress') {
    clearAllProgress().then(sendResponse);
    return true;
  } else if (request.action === 'getSettings') {
    getSettings().then(sendResponse);
    return true;
  } else if (request.action === 'updateSettings') {
    updateSettings(request.settings).then(sendResponse);
    return true;
  }
});

// Get all saved reading progress
async function getAllProgress () {
  try {
    const result = await chrome.storage.local.get(null);
    const progressEntries = [];
    const enabledPages = result.enabledPages || {};

    for (const [key, value] of Object.entries(result)) {
      if (key !== 'settings' && key !== 'enabledPages' && value.scrollPosition !== undefined) {
        progressEntries.push({
          url: key,
          isEnabled: enabledPages[key] || false,
          ...value
        });
      }
    }

    // Sort by timestamp (most recent first)
    progressEntries.sort((a, b) => b.timestamp - a.timestamp);

    return progressEntries;
  } catch (error) {
  // ...existing code...
    return [];
  }
}

// Clear all saved progress
async function clearAllProgress () {
  try {
    const result = await chrome.storage.local.get(null);
    const keysToRemove = Object.keys(result).filter(key => key !== 'settings' && key !== 'enabledPages');

    if (keysToRemove.length > 0) {
      await chrome.storage.local.remove(keysToRemove);
      return { success: true, count: keysToRemove.length };
    }

    return { success: true, count: 0 };
  } catch (error) {
  // ...existing code...
    return { success: false, error: error.message };
  }
}

// Get extension settings
async function getSettings () {
  try {
    const result = await chrome.storage.local.get('settings');
    return result.settings || {
      autoSave: true,
      showNotifications: true,
      maxSavedPages: 100
    };
  } catch (error) {
  // ...existing code...
    return null;
  }
}

// Update extension settings
async function updateSettings (newSettings) {
  try {
    await chrome.storage.local.set({ settings: newSettings });
    return { success: true };
  } catch (error) {
  // ...existing code...
    return { success: false, error: error.message };
  }
}

// Clean up old entries when storage limit is reached
chrome.storage.onChanged.addListener((changes, namespace) => {
  if (namespace === 'local') {
    // Check if we need to clean up old entries
    cleanupOldEntries();
  }
});

async function cleanupOldEntries () {
  try {
    const settings = await getSettings();
    if (!settings) return;

    const allProgress = await getAllProgress();

    if (allProgress.length > settings.maxSavedPages) {
      // Remove oldest entries
      const entriesToRemove = allProgress.slice(settings.maxSavedPages);
      const keysToRemove = entriesToRemove.map(entry => entry.url);

      await chrome.storage.local.remove(keysToRemove);
      // ...existing code...
    }
  } catch (error) {
  // ...existing code...
  }
}
