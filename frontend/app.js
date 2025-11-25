/**
 * Image Retrieval System - Frontend Application
 * Modern, interactive image search with AI-powered feedback
 */

// ============================================
// Configuration & State
// ============================================
const API_BASE = '';  // Same origin
const state = {
    sessionId: generateSessionId(),
    userId: 'user_' + Math.random().toString(36).substr(2, 9),
    currentModel: 'clip-base',
    topK: 20,
    results: [],
    likedImages: new Set(),
    dislikedImages: new Set(),
    searchType: 'text',
    isLoading: false
};

// ============================================
// Utility Functions
// ============================================
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function formatDate(isoString) {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Vừa xong';
    if (diffMins < 60) return `${diffMins} phút trước`;
    if (diffHours < 24) return `${diffHours} giờ trước`;
    if (diffDays < 7) return `${diffDays} ngày trước`;
    
    return date.toLocaleDateString('vi-VN');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-circle',
        info: 'fas fa-info-circle'
    };

    toast.innerHTML = `
        <i class="${icons[type]} toast-icon"></i>
        <span class="toast-message">${message}</span>
        <button class="toast-close"><i class="fas fa-times"></i></button>
    `;

    container.appendChild(toast);

    toast.querySelector('.toast-close').addEventListener('click', () => {
        toast.remove();
    });

    setTimeout(() => {
        if (toast.parentNode) {
            toast.remove();
        }
    }, 4000);
}

// ============================================
// API Functions
// ============================================
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        updateConnectionStatus(data.status === 'healthy');
        return data;
    } catch (error) {
        updateConnectionStatus(false);
        console.error('Health check failed:', error);
        return null;
    }
}

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        const data = await response.json();
        renderModels(data.models, data.default);
    } catch (error) {
        console.error('Failed to load models:', error);
        showToast('Không thể tải danh sách model', 'error');
    }
}

async function searchByText(query) {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: state.sessionId,
                query_text: query,
                top_k: state.topK,
                user_id: state.userId,
                model_key: state.currentModel
            })
        });

        if (!response.ok) throw new Error('Search failed');
        
        const data = await response.json();
        state.results = data.results;
        clearFeedback();
        renderResults(data.results);
        showToast(`Tìm thấy ${data.results.length} kết quả`, 'success');
    } catch (error) {
        console.error('Search error:', error);
        showToast('Có lỗi xảy ra khi tìm kiếm', 'error');
        showEmptyState();
    } finally {
        setLoading(false);
    }
}

async function searchByImage(file) {
    if (!file) return;

    setLoading(true);
    try {
        const formData = new FormData();
        formData.append('session_id', state.sessionId);
        formData.append('file', file);
        formData.append('top_k', state.topK);
        formData.append('user_id', state.userId);
        formData.append('model_key', state.currentModel);

        const response = await fetch(`${API_BASE}/search-by-image`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Image search failed');

        const data = await response.json();
        state.results = data.results;
        clearFeedback();
        renderResults(data.results);
        showToast(`Tìm thấy ${data.results.length} ảnh tương tự`, 'success');
    } catch (error) {
        console.error('Image search error:', error);
        showToast('Có lỗi xảy ra khi tìm kiếm bằng ảnh', 'error');
        showEmptyState();
    } finally {
        setLoading(false);
    }
}

async function submitFeedback() {
    const feedbackText = document.getElementById('feedbackText').value.trim();
    
    if (state.likedImages.size === 0 && state.dislikedImages.size === 0 && !feedbackText) {
        showToast('Vui lòng chọn ảnh thích/không thích hoặc nhập mô tả', 'info');
        return;
    }

    setLoading(true);
    try {
        const response = await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: state.sessionId,
                feedback_text: feedbackText || null,
                liked_image_ids: Array.from(state.likedImages),
                disliked_image_ids: Array.from(state.dislikedImages),
                w_text: 0.5,
                w_like: 0.5,
                alpha: 0.4,
                gamma: 0.5,
                top_k: state.topK,
                model_key: state.currentModel
            })
        });

        if (!response.ok) throw new Error('Feedback failed');

        const data = await response.json();
        state.results = data.results;
        clearFeedback();
        renderResults(data.results);
        document.getElementById('feedbackText').value = '';
        showToast('Kết quả đã được tinh chỉnh', 'success');
    } catch (error) {
        console.error('Feedback error:', error);
        showToast('Có lỗi xảy ra khi tinh chỉnh', 'error');
    } finally {
        setLoading(false);
    }
}

async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE}/history/${state.userId}?limit=20`);
        const data = await response.json();
        renderHistory(data.history || []);
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

async function loadStats() {
    try {
        const [statsRes, analyticsRes] = await Promise.all([
            fetch(`${API_BASE}/stats`),
            fetch(`${API_BASE}/analytics`)
        ]);

        const stats = await statsRes.json();
        const analytics = await analyticsRes.json();
        
        renderStats(stats, analytics);
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// ============================================
// UI Rendering Functions
// ============================================
function updateConnectionStatus(connected) {
    const indicator = document.getElementById('statusIndicator');
    indicator.classList.remove('connected', 'error');
    
    if (connected) {
        indicator.classList.add('connected');
        indicator.querySelector('.status-text').textContent = 'Đã kết nối';
    } else {
        indicator.classList.add('error');
        indicator.querySelector('.status-text').textContent = 'Lỗi kết nối';
    }
}

function renderModels(models, defaultModel) {
    const container = document.getElementById('modelOptions');
    container.innerHTML = models.map(model => `
        <div class="model-option ${model.key === defaultModel ? 'active' : ''}" data-model="${model.key}">
            <div class="model-option-header">
                <span class="model-option-name">${model.name}</span>
                <span class="model-option-check"><i class="fas fa-check"></i></span>
            </div>
            <p class="model-option-desc">${model.description}</p>
        </div>
    `).join('');

    state.currentModel = defaultModel;

    container.querySelectorAll('.model-option').forEach(option => {
        option.addEventListener('click', () => {
            container.querySelectorAll('.model-option').forEach(o => o.classList.remove('active'));
            option.classList.add('active');
            state.currentModel = option.dataset.model;
        });
    });
}

function renderResults(results) {
    const grid = document.getElementById('resultsGrid');
    const section = document.getElementById('resultsSection');
    const countEl = document.getElementById('resultsCount');

    if (results.length === 0) {
        showEmptyState();
        return;
    }

    section.classList.remove('hidden');
    document.getElementById('loadingState').classList.add('hidden');
    document.getElementById('emptyState').classList.add('hidden');
    
    countEl.textContent = `(${results.length})`;

    grid.innerHTML = results.map(result => {
        // Handle different path formats from database
        let imagePath = result.meta.file;
        if (imagePath.startsWith('data/')) {
            // Path already includes 'data/' prefix, just add leading slash
            imagePath = '/' + imagePath;
        } else if (imagePath.startsWith('/data/')) {
            // Path already complete
            imagePath = imagePath;
        } else if (imagePath.startsWith('full/')) {
            // Path starts with 'full/', add '/data/' prefix
            imagePath = '/data/' + imagePath;
        } else if (!imagePath.startsWith('/')) {
            // Relative path, add '/data/' prefix
            imagePath = '/data/' + imagePath;
        }
        const score = (result.score * 100).toFixed(1);
        const caption = result.meta.caption || 'Không có mô tả';
        const species = result.meta.species || 'Không xác định';

        return `
            <div class="result-card" data-id="${result.id}">
                <div class="result-image-container" onclick="openModal('${imagePath}', '${caption.replace(/'/g, "\\'")}', '${score}', '${species}')">
                    <img 
                        src="${imagePath}" 
                        alt="${caption}" 
                        class="result-image"
                        loading="lazy"
                        onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22200%22%3E%3Crect fill=%22%23f3f4f6%22 width=%22200%22 height=%22200%22/%3E%3Ctext fill=%22%239ca3af%22 font-family=%22sans-serif%22 font-size=%2214%22 x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22%3ENo Image%3C/text%3E%3C/svg%3E'"
                    >
                    <div class="result-overlay"></div>
                    <span class="result-score">${score}%</span>
                    <div class="result-actions">
                        <button class="feedback-btn like ${state.likedImages.has(result.id) ? 'active' : ''}" 
                                onclick="event.stopPropagation(); toggleLike('${result.id}')" 
                                title="Thích">
                            <i class="fas fa-heart"></i>
                        </button>
                        <button class="feedback-btn dislike ${state.dislikedImages.has(result.id) ? 'active' : ''}" 
                                onclick="event.stopPropagation(); toggleDislike('${result.id}')" 
                                title="Không thích">
                            <i class="fas fa-heart-broken"></i>
                        </button>
                    </div>
                </div>
                <div class="result-info">
                    <p class="result-caption" title="${caption}">${caption}</p>
                    <span class="result-species"><i class="fas fa-paw"></i> ${species}</span>
                </div>
            </div>
        `;
    }).join('');
}

function renderHistory(history) {
    const container = document.getElementById('historyList');
    
    if (history.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-history"></i>
                <h3>Chưa có lịch sử tìm kiếm</h3>
                <p>Hãy thực hiện tìm kiếm đầu tiên của bạn</p>
            </div>
        `;
        return;
    }

    container.innerHTML = history.map(item => `
        <div class="history-item">
            <div class="history-icon ${item.query_type}">
                <i class="fas fa-${item.query_type === 'text' ? 'keyboard' : 'image'}"></i>
            </div>
            <div class="history-content">
                <p class="history-query">${item.query_text || 'Không có nội dung'}</p>
                <div class="history-meta">
                    <span><i class="fas fa-clock"></i> ${formatDate(item.timestamp)}</span>
                    <span><i class="fas fa-images"></i> ${item.num_results} kết quả</span>
                </div>
            </div>
            <button class="history-action" onclick="reSearch('${item.query_text?.replace(/'/g, "\\'")}')">
                <i class="fas fa-redo"></i> Tìm lại
            </button>
        </div>
    `).join('');
}

function renderStats(stats, analytics) {
    // Main stats
    document.getElementById('totalImages').textContent = stats.total_images?.toLocaleString() || 0;
    document.getElementById('totalSearches').textContent = analytics.total_searches?.toLocaleString() || 0;
    document.getElementById('totalUsers').textContent = analytics.total_users?.toLocaleString() || 0;
    document.getElementById('totalModels').textContent = Object.keys(stats.models || {}).length;

    // Model stats
    const modelStatsContainer = document.getElementById('modelStats');
    const models = stats.models || {};
    modelStatsContainer.innerHTML = Object.entries(models).map(([key, info]) => `
        <div class="model-stat-item">
            <span class="model-stat-name">${info.model_name || key}</span>
            <span class="model-stat-count">${info.count?.toLocaleString() || 0} ảnh</span>
        </div>
    `).join('') || '<p style="color: var(--gray-500);">Không có dữ liệu</p>';

    // Search types chart
    const textCount = analytics.query_types?.text || 0;
    const imageCount = analytics.query_types?.image || 0;
    const total = textCount + imageCount;
    const textPercent = total > 0 ? Math.round((textCount / total) * 100) : 50;

    document.getElementById('textSearchCount').textContent = textCount;
    document.getElementById('imageSearchCount').textContent = imageCount;
    document.getElementById('textSearchPercent').textContent = `${textPercent}%`;

    const chart = document.getElementById('searchTypesChart');
    chart.style.setProperty('--text-percent', `${textPercent * 3.6}deg`);

    // Top queries
    const topQueriesContainer = document.getElementById('topQueries');
    const topQueries = analytics.top_queries || [];
    
    if (topQueries.length === 0) {
        topQueriesContainer.innerHTML = '<p style="color: var(--gray-500); text-align: center; padding: 1rem;">Chưa có dữ liệu</p>';
    } else {
        topQueriesContainer.innerHTML = topQueries.map((item, index) => `
            <div class="query-item">
                <span class="query-rank">${index + 1}</span>
                <span class="query-text">${item.query}</span>
                <span class="query-count">${item.count} lượt</span>
            </div>
        `).join('');
    }
}

function setLoading(loading) {
    state.isLoading = loading;
    const loadingState = document.getElementById('loadingState');
    const resultsGrid = document.getElementById('resultsGrid');
    
    if (loading) {
        document.getElementById('resultsSection').classList.remove('hidden');
        loadingState.classList.remove('hidden');
        resultsGrid.innerHTML = '';
        document.getElementById('emptyState').classList.add('hidden');
    } else {
        loadingState.classList.add('hidden');
    }
}

function showEmptyState() {
    document.getElementById('resultsSection').classList.remove('hidden');
    document.getElementById('loadingState').classList.add('hidden');
    document.getElementById('emptyState').classList.remove('hidden');
    document.getElementById('resultsGrid').innerHTML = '';
    document.getElementById('resultsCount').textContent = '(0)';
}

// ============================================
// Feedback Functions
// ============================================
function toggleLike(imageId) {
    if (state.likedImages.has(imageId)) {
        state.likedImages.delete(imageId);
    } else {
        state.likedImages.add(imageId);
        state.dislikedImages.delete(imageId);
    }
    updateFeedbackUI();
    updateCardFeedbackUI(imageId);
}

function toggleDislike(imageId) {
    if (state.dislikedImages.has(imageId)) {
        state.dislikedImages.delete(imageId);
    } else {
        state.dislikedImages.add(imageId);
        state.likedImages.delete(imageId);
    }
    updateFeedbackUI();
    updateCardFeedbackUI(imageId);
}

function updateCardFeedbackUI(imageId) {
    const card = document.querySelector(`.result-card[data-id="${imageId}"]`);
    if (!card) return;

    const likeBtn = card.querySelector('.feedback-btn.like');
    const dislikeBtn = card.querySelector('.feedback-btn.dislike');

    likeBtn.classList.toggle('active', state.likedImages.has(imageId));
    dislikeBtn.classList.toggle('active', state.dislikedImages.has(imageId));
}

function updateFeedbackUI() {
    document.getElementById('likedCount').textContent = state.likedImages.size;
    document.getElementById('dislikedCount').textContent = state.dislikedImages.size;
    
    const feedbackBtn = document.getElementById('submitFeedbackBtn');
    const feedbackText = document.getElementById('feedbackText').value.trim();
    feedbackBtn.disabled = state.likedImages.size === 0 && state.dislikedImages.size === 0 && !feedbackText;
}

function clearFeedback() {
    state.likedImages.clear();
    state.dislikedImages.clear();
    updateFeedbackUI();
}

// ============================================
// Modal Functions
// ============================================
function openModal(imageSrc, caption, score, species) {
    const modal = document.getElementById('imageModal');
    document.getElementById('modalImage').src = imageSrc;
    document.getElementById('modalCaption').textContent = caption;
    document.getElementById('modalScore').textContent = score + '%';
    document.getElementById('modalSpecies').textContent = species;
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    document.getElementById('imageModal').classList.add('hidden');
    document.body.style.overflow = '';
}

// ============================================
// Helper Functions
// ============================================
function reSearch(query) {
    if (!query || query === 'null') return;
    
    // Switch to search tab
    switchTab('search');
    
    // Set query and search
    document.getElementById('searchQuery').value = query;
    searchByText(query);
}

function switchTab(tabName) {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.tab === tabName);
    });
    
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.toggle('active', tab.id === `${tabName}Tab`);
    });

    // Load data for specific tabs
    if (tabName === 'history') {
        loadHistory();
    } else if (tabName === 'analytics') {
        loadStats();
    }
}

// ============================================
// Event Listeners
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    // Initialize
    checkHealth();
    loadModels();

    // Navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            switchTab(link.dataset.tab);
        });
    });

    // Search type toggle
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const type = btn.dataset.type;
            state.searchType = type;
            
            document.getElementById('textSearchForm').classList.toggle('hidden', type !== 'text');
            document.getElementById('imageSearchForm').classList.toggle('hidden', type !== 'image');
        });
    });

    // Text search form
    document.getElementById('textSearchForm').addEventListener('submit', (e) => {
        e.preventDefault();
        const query = document.getElementById('searchQuery').value;
        searchByText(query);
    });

    // Suggestion chips
    document.querySelectorAll('.suggestion-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.dataset.query;
            document.getElementById('searchQuery').value = query;
            searchByText(query);
        });
    });

    // Image upload
    const imageInput = document.getElementById('imageInput');
    const uploadZone = document.getElementById('imageUploadZone');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const imageSearchBtn = document.getElementById('imageSearchBtn');

    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                imagePreview.classList.remove('hidden');
                uploadZone.querySelector('.upload-content').style.display = 'none';
                imageSearchBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    });

    document.getElementById('removeImageBtn').addEventListener('click', () => {
        imageInput.value = '';
        imagePreview.classList.add('hidden');
        uploadZone.querySelector('.upload-content').style.display = '';
        imageSearchBtn.disabled = true;
    });

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            imageInput.files = e.dataTransfer.files;
            imageInput.dispatchEvent(new Event('change'));
        }
    });

    // Image search form
    document.getElementById('imageSearchForm').addEventListener('submit', (e) => {
        e.preventDefault();
        const file = imageInput.files[0];
        if (file) {
            searchByImage(file);
        }
    });

    // Advanced options
    document.getElementById('advancedToggle').addEventListener('click', function() {
        this.classList.toggle('open');
        document.getElementById('advancedPanel').classList.toggle('hidden');
    });

    document.getElementById('topKSlider').addEventListener('input', function() {
        document.getElementById('topKValue').textContent = this.value;
        state.topK = parseInt(this.value);
    });

    // Feedback
    document.getElementById('submitFeedbackBtn').addEventListener('click', submitFeedback);
    
    document.getElementById('clearFeedbackBtn').addEventListener('click', () => {
        clearFeedback();
        document.getElementById('feedbackText').value = '';
        renderResults(state.results);
    });

    document.getElementById('feedbackText').addEventListener('input', updateFeedbackUI);

    // Modal
    document.getElementById('modalClose').addEventListener('click', closeModal);
    document.querySelector('.modal-overlay').addEventListener('click', closeModal);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });

    // Refresh buttons
    document.getElementById('refreshHistoryBtn').addEventListener('click', loadHistory);
    document.getElementById('refreshAnalyticsBtn').addEventListener('click', loadStats);
});

// Make functions globally accessible for inline handlers
window.toggleLike = toggleLike;
window.toggleDislike = toggleDislike;
window.openModal = openModal;
window.reSearch = reSearch;
