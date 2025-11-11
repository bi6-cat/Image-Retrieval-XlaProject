const API_BASE = "http://localhost:8000";
const session_id = "session-" + Math.random().toString(36).substring(2, 9);
const user_id = localStorage.getItem('user_id') || "user-" + Math.random().toString(36).substring(2, 9);
localStorage.setItem('user_id', user_id);

let likedImages = [];
let dislikedImages = [];
let currentTab = 'text';
let uploadedFile = null;

// Initialize
window.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadHistory();
});

document.getElementById("searchBtn").addEventListener("click", searchImages);
document.getElementById("imageUpload").addEventListener("change", handleFileSelect);
document.getElementById("searchByImageBtn").addEventListener("click", searchByImage);
document.getElementById("manualRefineBtn").addEventListener("click", manualRefineWithText);
document.getElementById("visualRefineBtn").addEventListener("click", visualRefine);

// Enter key to search
document.getElementById("queryInput").addEventListener("keypress", (e) => {
  if (e.key === 'Enter') searchImages();
});

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.search-panel').forEach(panel => panel.classList.remove('active'));
  
  if (tab === 'text') {
    document.querySelector('.tab-btn:first-child').classList.add('active');
    document.getElementById('textSearch').classList.add('active');
  } else {
    document.querySelector('.tab-btn:last-child').classList.add('active');
    document.getElementById('imageSearch').classList.add('active');
  }
}

function handleFileSelect(e) {
  uploadedFile = e.target.files[0];
  if (uploadedFile) {
    document.getElementById('fileName').textContent = uploadedFile.name;
    document.getElementById('searchByImageBtn').style.display = 'inline-block';
  }
}

async function searchByImage() {
  if (!uploadedFile) return alert("Please select an image first!");
  
  showLoading(true);
  const formData = new FormData();
  formData.append('file', uploadedFile);
  formData.append('session_id', session_id);
  formData.append('user_id', user_id);
  formData.append('top_k', '12');

  try {
    const res = await fetch(`${API_BASE}/search-by-image`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error('Search failed');
    const data = await res.json();
    renderResults(data.results, `image: ${uploadedFile.name}`);
    loadHistory();  // Refresh history
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    showLoading(false);
  }
}

async function searchImages() {
  const query = document.getElementById("queryInput").value.trim();
  if (!query) return alert("Please enter a query!");

  showLoading(true);
  try {
    const res = await fetch(`${API_BASE}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        session_id, 
        query_text: query, 
        top_k: 12,
        user_id: user_id
      })
    });

    if (!res.ok) throw new Error('Search failed');
    const data = await res.json();
    renderResults(data.results, `"${query}"`);
    loadHistory();  // Refresh history
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    showLoading(false);
  }
}

function showLoading(show) {
  document.getElementById('loading').style.display = show ? 'flex' : 'none';
}

function renderResults(results, query, preserveFeedback = false) {
  const container = document.getElementById("results");
  const info = document.getElementById("resultsInfo");
  container.innerHTML = "";
  
  // Only reset feedback if not preserving (i.e., new search)
  if (!preserveFeedback) {
    likedImages = [];
    dislikedImages = [];
  }

  info.innerHTML = `Found <strong>${results.length}</strong> results for ${query}`;

  results.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "image-card";
    card.dataset.imageId = item.id;
    
    console.log(`Rendering card ${idx}: id=${item.id}, species=${item.meta?.species}`);
    
    const img = document.createElement("img");
    img.src = item.meta?.file || "https://via.placeholder.com/200";
    img.alt = item.meta?.species || "Animal";
    img.loading = "lazy";
    img.onclick = () => openModal(img.src);
    
    // Optional: Add metadata overlay on hover (very subtle)
    const overlay = document.createElement("div");
    overlay.className = "card-overlay";
    
    let tagsHtml = '';
    if (item.meta?.color) {
      tagsHtml += `<span class="tag tag-color">${item.meta.color}</span>`;
    }
    if (item.meta?.action) {
      tagsHtml += `<span class="tag tag-action">${item.meta.action}</span>`;
    }
    if (item.meta?.environment) {
      tagsHtml += `<span class="tag tag-env">${item.meta.environment}</span>`;
    }
    
    overlay.innerHTML = `
      ${item.meta?.caption ? `<div class="caption-overlay">${item.meta.caption}</div>` : ''}
      ${tagsHtml ? `<div class="tags-overlay">${tagsHtml}</div>` : ''}
    `;
    
    const feedbackBtns = document.createElement("div");
    feedbackBtns.className = "feedback-buttons";
    
    const likeBtn = document.createElement("button");
    likeBtn.className = "feedback-btn like-btn";
    likeBtn.innerHTML = "👍 Relevant";
    likeBtn.onclick = (e) => {
      e.stopPropagation();
      toggleFeedback(item.id, 'like', card);
    };
    
    const dislikeBtn = document.createElement("button");
    dislikeBtn.className = "feedback-btn dislike-btn";
    dislikeBtn.innerHTML = "� Not Relevant";
    dislikeBtn.onclick = (e) => {
      e.stopPropagation();
      toggleFeedback(item.id, 'dislike', card);
    };
    
    feedbackBtns.appendChild(likeBtn);
    feedbackBtns.appendChild(dislikeBtn);
    
    card.appendChild(img);
    card.appendChild(overlay);
    card.appendChild(feedbackBtns);
    
    // Restore feedback state if preserving
    if (preserveFeedback) {
      if (likedImages.includes(item.id)) {
        card.classList.add('liked');
      } else if (dislikedImages.includes(item.id)) {
        card.classList.add('disliked');
      }
    }
    
    container.appendChild(card);
  });

  updateFeedbackStats();
}

function toggleFeedback(id, type, card) {
  console.log(`toggleFeedback called: id=${id}, type=${type}`);
  console.log(`Before: liked=${likedImages.length}, disliked=${dislikedImages.length}`);
  
  if (type === 'like') {
    // If already liked, remove like (toggle off)
    if (likedImages.includes(id)) {
      likedImages = likedImages.filter(x => x !== id);
      card.classList.remove('liked');
      console.log(`Removed from liked`);
    } else {
      // Remove from dislike if was disliked
      dislikedImages = dislikedImages.filter(x => x !== id);
      card.classList.remove('disliked');
      // Add to liked
      likedImages.push(id);
      card.classList.add('liked');
      console.log(`Added to liked`);
    }
  } else if (type === 'dislike') {
    // If already disliked, remove dislike (toggle off)
    if (dislikedImages.includes(id)) {
      dislikedImages = dislikedImages.filter(x => x !== id);
      card.classList.remove('disliked');
      console.log(`Removed from disliked`);
    } else {
      // Remove from like if was liked
      likedImages = likedImages.filter(x => x !== id);
      card.classList.remove('liked');
      // Add to disliked
      dislikedImages.push(id);
      card.classList.add('disliked');
      console.log(`Added to disliked`);
    }
  }
  
  console.log(`After: liked=${likedImages.length}, disliked=${dislikedImages.length}`);
  console.log(`Liked IDs:`, likedImages);
  console.log(`Disliked IDs:`, dislikedImages);
  
  updateFeedbackStats();
  
  // Remove auto-refine - user will click button instead
}

function updateFeedbackStats() {
  const stats = document.getElementById('feedbackStats');
  const totalFeedback = likedImages.length + dislikedImages.length;
  
  let html = '';
  if (likedImages.length > 0) {
    html += `<span class="stat-badge liked-badge">✅ ${likedImages.length} Relevant</span>`;
  }
  if (dislikedImages.length > 0) {
    html += `<span class="stat-badge disliked-badge">❌ ${dislikedImages.length} Not Relevant</span>`;
  }
  
  if (totalFeedback === 0) {
    html = '<span class="hint-text">👆 Mark images as Relevant or Not Relevant above</span>';
  }
  
  stats.innerHTML = html;
}

// Auto-refine function removed - now manual button only

// Visual refine (like/dislike only)
async function visualRefine() {
  if (likedImages.length === 0 && dislikedImages.length === 0) {
    return alert('Please mark some images as Relevant or Not Relevant first!');
  }
  
  console.log('Visual refine:', { likedImages, dislikedImages, session_id });
  showFeedbackLoading(true);
  
  try {
    const res = await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: session_id,
        feedback_text: null,
        liked_image_ids: likedImages || [],
        disliked_image_ids: dislikedImages || [],
        alpha: 0.4,
        gamma: 0.5,
        top_k: 12
      })
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));
      console.error('Refine error:', errorData);
      throw new Error(errorData.detail || 'Refine failed');
    }
    const data = await res.json();
    renderResults(data.results, 'refined search', true);  // Preserve feedback state
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    showFeedbackLoading(false);
  }
}

// Manual refine with text feedback
async function manualRefineWithText() {
  const feedbackText = document.getElementById('feedbackText').value.trim();
  
  if (!feedbackText && likedImages.length === 0 && dislikedImages.length === 0) {
    return alert('Please provide text feedback or mark some images!');
  }
  
  showFeedbackLoading(true);
  
  try {
    const res = await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id,
        feedback_text: feedbackText || null,
        liked_image_ids: likedImages,
        disliked_image_ids: dislikedImages,
        alpha: 0.4,
        gamma: 0.5,
        w_text: 0.6,  // Higher weight for text when manually provided
        w_like: 0.4,
        top_k: 12
      })
    });

    if (!res.ok) throw new Error('Refine failed');
    const data = await res.json();
    renderResults(data.results, feedbackText ? `refined: "${feedbackText}"` : 'refined search', true);  // Preserve feedback state
    
    // Clear text after successful refine
    document.getElementById('feedbackText').value = '';
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    showFeedbackLoading(false);
  }
}

// Show loading state specifically for feedback
function showFeedbackLoading(show) {
  const stats = document.getElementById('feedbackStats');
  if (show) {
    const loadingBadge = document.createElement('span');
    loadingBadge.className = 'stat-badge loading-badge';
    loadingBadge.innerHTML = '⏳ Refining results...';
    loadingBadge.id = 'loadingBadge';
    stats.appendChild(loadingBadge);
  } else {
    const loadingBadge = document.getElementById('loadingBadge');
    if (loadingBadge) {
      loadingBadge.remove();
    }
  }
}

function openModal(src) {
  const modal = document.createElement('div');
  modal.className = 'modal';
  modal.innerHTML = `
    <div class="modal-content">
      <span class="modal-close" onclick="this.parentElement.parentElement.remove()">×</span>
      <img src="${src}" />
    </div>
  `;
  modal.onclick = (e) => {
    if (e.target === modal) modal.remove();
  };
  document.body.appendChild(modal);
}

window.addEventListener("beforeunload", () => {
  navigator.sendBeacon(`${API_BASE}/session/end`, JSON.stringify({
    session_id,
    user_id,
    delta: 0.05
  }));
});

// Stats and History functions
async function loadStats() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    const data = await res.json();
    document.getElementById('statsBar').innerHTML = `
      <span>📊 Total Images: <strong>${data.total_images}</strong></span>
      <span>👤 User: <strong>${user_id.substring(0, 10)}...</strong></span>
    `;
  } catch (err) {
    console.error('Failed to load stats:', err);
  }
}

async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/history/${user_id}?limit=10`);
    const data = await res.json();
    const historyList = document.getElementById('historyList');
    
    if (data.history && data.history.length > 0) {
      historyList.innerHTML = data.history.map((entry, idx) => `
        <div class="history-item" onclick="repeatSearch('${entry.query_text}', '${entry.query_type}')">
          <span class="history-icon">${entry.query_type === 'image' ? '📸' : '🔤'}</span>
          <div class="history-details">
            <div class="history-query">${entry.query_text}</div>
            <div class="history-meta">
              ${new Date(entry.timestamp).toLocaleString()} • ${entry.num_results} results
            </div>
          </div>
        </div>
      `).join('');
    } else {
      historyList.innerHTML = '<p class="hint-text">No search history yet</p>';
    }
  } catch (err) {
    console.error('Failed to load history:', err);
  }
}

function toggleHistory() {
  const historyList = document.getElementById('historyList');
  const btn = document.getElementById('toggleHistory');
  if (historyList.style.display === 'none') {
    historyList.style.display = 'block';
    btn.textContent = 'Hide History';
  } else {
    historyList.style.display = 'none';
    btn.textContent = 'Show History';
  }
}

function repeatSearch(query, type) {
  if (type === 'text' && !query.startsWith('[Image:')) {
    document.getElementById('queryInput').value = query;
    searchImages();
  }
}
