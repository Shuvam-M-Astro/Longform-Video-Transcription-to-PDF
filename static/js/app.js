/**
 * Video Documentation Builder - Frontend JavaScript
 * Handles form submissions, WebSocket communication, and real-time updates
 */

class VideoProcessor {
    constructor() {
        this.socket = null;
        this.currentJobId = null;
        this.isProcessing = false;
        this.lastSearchResults = null; // Store last search results for export
        this.currentSearchPage = 1; // Current search results page
        this.searchPerPage = 20; // Results per page
        this.lastSearchQuery = null; // Track last search query to detect new searches
        this.currentBatchId = null; // Current batch job ID
        this.batchStatusInterval = null; // Interval for batch status updates
        this.init();
    }

    init() {
        this.setupSocketConnection();
        this.setupEventListeners();
        this.loadRecentJobs();
        this.loadPresets();
    }

    setupSocketConnection() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.showNotification('Connected to processing server', 'success');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.showNotification('Disconnected from server', 'warning');
        });

        this.socket.on('error', (data) => {
            console.error('WebSocket error:', data);
            this.showNotification(`Connection error: ${data.message}`, 'error');
        });

        this.socket.on('job_status', (data) => {
            this.handleJobStatusUpdate(data);
        });

        this.socket.on('job_progress', (data) => {
            this.handleProgressUpdate(data);
        });

        this.socket.on('job_step', (data) => {
            this.handleStepUpdate(data);
        });
    }

    setupEventListeners() {
        // File upload form
        const fileForm = document.getElementById('fileUploadForm');
        if (fileForm) {
            fileForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleFileUpload();
            });
        }

        // URL form
        const urlForm = document.getElementById('urlForm');
        if (urlForm) {
            urlForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleUrlProcessing();
            });
        }

        // Search form
        const searchForm = document.getElementById('searchForm');
        if (searchForm) {
            searchForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleSearch(true); // Reset to page 1 for new searches
            });
        }

        // Export buttons
        document.getElementById('exportResultsCsv')?.addEventListener('click', () => {
            this.exportSearchResults('csv');
        });

        document.getElementById('exportResultsJson')?.addEventListener('click', () => {
            this.exportSearchResults('json');
        });

        // Download buttons
        document.getElementById('downloadPdf')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.downloadFile('pdf');
        });

        document.getElementById('downloadTranscript')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.downloadFile('transcript');
        });

        document.getElementById('downloadAudio')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.downloadFile('audio');
        });

        // Batch processing
        document.getElementById('startBatchBtn')?.addEventListener('click', () => {
            this.startBatchProcessing();
        });

        document.getElementById('cancelBatchBtn')?.addEventListener('click', () => {
            this.cancelBatch();
        });

        document.getElementById('batchFiles')?.addEventListener('change', (e) => {
            this.handleBatchFilesChange(e);
        });

        // Tab switching
        const tabs = document.querySelectorAll('[data-bs-toggle="tab"]');
        tabs.forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                this.updateFormValidation(e.target);
            });
        });

        // Streaming checkbox (only for URL tab)
        const streamingCheckbox = document.getElementById('streaming');
        const urlTab = document.getElementById('url-tab');
        if (streamingCheckbox && urlTab) {
            urlTab.addEventListener('click', () => {
                streamingCheckbox.disabled = false;
            });
            
            // Disable streaming for file upload tab
            document.getElementById('file-tab')?.addEventListener('click', () => {
                streamingCheckbox.disabled = true;
                streamingCheckbox.checked = false;
            });
        }
    }

    async handleFileUpload() {
        if (this.isProcessing) {
            this.showNotification('Already processing a video. Please wait.', 'warning');
            return;
        }

        const fileInput = document.getElementById('fileInput');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showNotification('Please select a video file', 'error');
            return;
        }

        // Validate file size (2GB limit)
        const maxSize = 2 * 1024 * 1024 * 1024; // 2GB
        if (file.size > maxSize) {
            this.showNotification('File size exceeds 2GB limit', 'error');
            return;
        }

        // Validate file type
        const allowedTypes = [
            'video/mp4', 'video/mkv', 'video/mov', 'video/webm',
            'audio/mp3', 'audio/m4a', 'audio/wav', 'audio/flac'
        ];
        
        if (!allowedTypes.includes(file.type) && !this.isVideoFile(file.name)) {
            this.showNotification('Unsupported file type. Please select a video or audio file.', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        
        // Add processing options
        const options = this.getProcessingOptions();
        Object.keys(options).forEach(key => {
            formData.append(key, options[key]);
        });

        try {
            this.showProcessingCard();
            this.isProcessing = true;
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (response.ok) {
                this.currentJobId = result.job_id;
                this.showNotification('File uploaded successfully. Processing started.', 'success');
                this.addToProcessingLog('File uploaded and processing started');
            } else {
                // Handle different error types
                if (response.status === 413) {
                    throw new Error('File too large. Please choose a smaller file.');
                } else if (response.status === 400) {
                    throw new Error(result.error || 'Invalid file or options');
                } else if (response.status === 500) {
                    throw new Error('Server error. Please try again later.');
                } else {
                    throw new Error(result.error || 'Upload failed');
                }
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showNotification(`Upload failed: ${error.message}`, 'error');
            this.hideProcessingCard();
            this.isProcessing = false;
        }
    }

    async handleUrlProcessing() {
        if (this.isProcessing) {
            this.showNotification('Already processing a video. Please wait.', 'warning');
            return;
        }

        const urlInput = document.getElementById('urlInput');
        const url = urlInput.value.trim();
        
        if (!url) {
            this.showNotification('Please enter a video URL', 'error');
            return;
        }

        // Basic URL validation
        try {
            new URL(url);
        } catch {
            this.showNotification('Please enter a valid URL', 'error');
            return;
        }

        const options = this.getProcessingOptions();
        const requestData = {
            url: url,
            ...options
        };

        try {
            this.showProcessingCard();
            this.isProcessing = true;
            
            const response = await fetch('/process_url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();
            
            if (response.ok) {
                this.currentJobId = result.job_id;
                this.showNotification('URL processing started successfully.', 'success');
                this.addToProcessingLog('URL processing started');
            } else {
                // Handle different error types
                if (response.status === 400) {
                    throw new Error(result.error || 'Invalid URL or options');
                } else if (response.status === 500) {
                    throw new Error('Server error. Please try again later.');
                } else {
                    throw new Error(result.error || 'Processing failed');
                }
            }
        } catch (error) {
            console.error('URL processing error:', error);
            this.showNotification(`Processing failed: ${error.message}`, 'error');
            this.hideProcessingCard();
            this.isProcessing = false;
        }
    }

    getProcessingOptions() {
        return {
            language: document.getElementById('language').value,
            whisper_model: document.getElementById('whisperModel').value,
            beam_size: parseInt(document.getElementById('beamSize')?.value || 5),
            transcribe_only: document.getElementById('transcribeOnly').checked,
            streaming: document.getElementById('streaming').checked,
            kf_method: document.getElementById('kfMethod').value,
            max_fps: parseFloat(document.getElementById('maxFps')?.value || 1.0),
            min_scene_diff: parseFloat(document.getElementById('minSceneDiff')?.value || 0.45),
            report_style: document.getElementById('reportStyle').value
        };
    }

    async loadPresets() {
        try {
            const response = await fetch('/api/presets');
            if (response.ok) {
                const data = await response.json();
                const selector = document.getElementById('presetSelector');
                if (selector) {
                    // Clear existing options except the first one
                    selector.innerHTML = '<option value="">Load Preset...</option>';
                    
                    data.presets.forEach(preset => {
                        const option = document.createElement('option');
                        option.value = preset.id;
                        option.textContent = preset.name + (preset.is_default ? ' (Default)' : '');
                        if (preset.is_public) {
                            option.textContent += ' 🌐';
                        }
                        selector.appendChild(option);
                    });
                }
            }
        } catch (error) {
            console.error('Failed to load presets:', error);
        }
    }

    async loadPreset(presetId) {
        if (!presetId) return;
        
        try {
            const response = await fetch(`/api/presets/${presetId}`);
            if (response.ok) {
                const preset = await response.json();
                this.applyPresetConfig(preset.config);
                
                // Record usage
                await fetch(`/api/presets/${presetId}/use`, { method: 'POST' });
                
                this.showNotification(`Loaded preset: ${preset.name}`, 'success');
            } else {
                throw new Error('Failed to load preset');
            }
        } catch (error) {
            this.showNotification(`Failed to load preset: ${error.message}`, 'error');
        }
    }

    applyPresetConfig(config) {
        if (config.language) document.getElementById('language').value = config.language;
        if (config.whisper_model) document.getElementById('whisperModel').value = config.whisper_model;
        if (config.beam_size !== undefined) {
            const beamSizeInput = document.getElementById('beamSize');
            if (beamSizeInput) beamSizeInput.value = config.beam_size;
        }
        if (config.transcribe_only !== undefined) document.getElementById('transcribeOnly').checked = config.transcribe_only;
        if (config.streaming !== undefined) document.getElementById('streaming').checked = config.streaming;
        if (config.kf_method) document.getElementById('kfMethod').value = config.kf_method;
        if (config.max_fps !== undefined) {
            const maxFpsInput = document.getElementById('maxFps');
            if (maxFpsInput) maxFpsInput.value = config.max_fps;
        }
        if (config.min_scene_diff !== undefined) {
            const minSceneDiffInput = document.getElementById('minSceneDiff');
            if (minSceneDiffInput) minSceneDiffInput.value = config.min_scene_diff;
        }
        if (config.report_style) document.getElementById('reportStyle').value = config.report_style;
        
        // Reset selector
        const selector = document.getElementById('presetSelector');
        if (selector) selector.value = '';
    }

    async saveCurrentAsPreset() {
        const name = prompt('Enter a name for this preset:');
        if (!name || !name.trim()) return;
        
        const description = prompt('Enter a description (optional):') || '';
        const isDefault = confirm('Set as default preset?');
        const isPublic = confirm('Make this preset public (shareable)?');
        
        const config = this.getProcessingOptions();
        
        try {
            const response = await fetch('/api/presets', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: name.trim(),
                    description: description.trim(),
                    config: config,
                    is_default: isDefault,
                    is_public: isPublic
                })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showNotification('Preset saved successfully', 'success');
                await this.loadPresets();
            } else {
                throw new Error(result.error || 'Failed to save preset');
            }
        } catch (error) {
            this.showNotification(`Failed to save preset: ${error.message}`, 'error');
        }
    }

    async managePresets() {
        // Simple modal for managing presets
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Manage Presets</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div id="presetsList"></div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
        
        // Load presets list
        try {
            const response = await fetch('/api/presets');
            if (response.ok) {
                const data = await response.json();
                const listContainer = document.getElementById('presetsList');
                
                if (data.presets.length === 0) {
                    listContainer.innerHTML = '<p class="text-muted">No presets saved yet.</p>';
                } else {
                    listContainer.innerHTML = data.presets.map(preset => `
                        <div class="card mb-2">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-start">
                                    <div>
                                        <h6 class="mb-1">
                                            ${this.escapeHtml(preset.name)}
                                            ${preset.is_default ? '<span class="badge bg-primary ms-2">Default</span>' : ''}
                                            ${preset.is_public ? '<span class="badge bg-info ms-2">Public</span>' : ''}
                                        </h6>
                                        ${preset.description ? `<p class="text-muted small mb-1">${this.escapeHtml(preset.description)}</p>` : ''}
                                        <small class="text-muted">Used ${preset.usage_count} times</small>
                                    </div>
                                    <div class="btn-group btn-group-sm">
                                        <button class="btn btn-outline-primary" onclick="videoProcessor.loadPreset('${preset.id}'); bootstrap.Modal.getInstance(document.querySelector('.modal.show')).hide();">
                                            <i class="fas fa-download"></i> Load
                                        </button>
                                        ${preset.is_owner ? `
                                            <button class="btn btn-outline-danger" onclick="videoProcessor.deletePreset('${preset.id}')">
                                                <i class="fas fa-trash"></i>
                                            </button>
                                        ` : ''}
                                    </div>
                                </div>
                            </div>
                        </div>
                    `).join('');
                }
            }
        } catch (error) {
            console.error('Failed to load presets:', error);
        }
        
        // Clean up on close
        modal.addEventListener('hidden.bs.modal', () => {
            document.body.removeChild(modal);
        });
    }

    async deletePreset(presetId) {
        if (!confirm('Are you sure you want to delete this preset?')) return;
        
        try {
            const response = await fetch(`/api/presets/${presetId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                this.showNotification('Preset deleted successfully', 'success');
                await this.loadPresets();
                await this.managePresets(); // Refresh the modal
            } else {
                const result = await response.json();
                throw new Error(result.error || 'Failed to delete preset');
            }
        } catch (error) {
            this.showNotification(`Failed to delete preset: ${error.message}`, 'error');
        }
    }

    async showWebhooksModal() {
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Webhook Notifications</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h6 class="mb-0">Configured Webhooks</h6>
                            <button class="btn btn-primary btn-sm" onclick="videoProcessor.showCreateWebhookForm()">
                                <i class="fas fa-plus"></i> Add Webhook
                            </button>
                        </div>
                        <div id="webhooksList"></div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
        
        await this.loadWebhooks();
        
        modal.addEventListener('hidden.bs.modal', () => {
            document.body.removeChild(modal);
        });
    }

    async loadWebhooks() {
        try {
            const response = await fetch('/api/webhooks');
            if (response.ok) {
                const data = await response.json();
                const listContainer = document.getElementById('webhooksList');
                
                if (data.webhooks.length === 0) {
                    listContainer.innerHTML = '<p class="text-muted">No webhooks configured. Click "Add Webhook" to create one.</p>';
                } else {
                    listContainer.innerHTML = data.webhooks.map(webhook => `
                        <div class="card mb-2">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-start">
                                    <div class="flex-grow-1">
                                        <h6 class="mb-1">
                                            ${this.escapeHtml(webhook.name)}
                                            ${webhook.is_active ? '<span class="badge bg-success ms-2">Active</span>' : '<span class="badge bg-secondary ms-2">Inactive</span>'}
                                        </h6>
                                        <p class="text-muted small mb-1"><code>${this.escapeHtml(webhook.url)}</code></p>
                                        <div class="small">
                                            <strong>Events:</strong> ${(webhook.events || []).join(', ')}<br>
                                            <strong>Stats:</strong> ${webhook.success_count} success, ${webhook.failure_count} failed
                                        </div>
                                    </div>
                                    <div class="btn-group btn-group-sm">
                                        <button class="btn btn-outline-primary" onclick="videoProcessor.toggleWebhook('${webhook.id}')" title="${webhook.is_active ? 'Deactivate' : 'Activate'}">
                                            <i class="fas fa-${webhook.is_active ? 'pause' : 'play'}"></i>
                                        </button>
                                        <button class="btn btn-outline-danger" onclick="videoProcessor.deleteWebhook('${webhook.id}')" title="Delete">
                                            <i class="fas fa-trash"></i>
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `).join('');
                }
            }
        } catch (error) {
            console.error('Failed to load webhooks:', error);
        }
    }

    showCreateWebhookForm() {
        const name = prompt('Webhook name:');
        if (!name) return;
        
        const url = prompt('Webhook URL:');
        if (!url) return;
        
        const eventsInput = prompt('Events (comma-separated: job.started, job.completed, job.failed):', 'job.completed, job.failed');
        const events = eventsInput ? eventsInput.split(',').map(e => e.trim()) : ['job.completed'];
        
        const secret = prompt('Secret (optional, for HMAC signature):') || '';
        
        this.createWebhook({
            name: name,
            url: url,
            events: events,
            secret: secret
        });
    }

    async createWebhook(data) {
        try {
            const response = await fetch('/api/webhooks', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showNotification('Webhook created successfully', 'success');
                await this.loadWebhooks();
            } else {
                throw new Error(result.error || 'Failed to create webhook');
            }
        } catch (error) {
            this.showNotification(`Failed to create webhook: ${error.message}`, 'error');
        }
    }

    async toggleWebhook(webhookId) {
        try {
            const response = await fetch(`/api/webhooks/${webhookId}/toggle`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.showNotification('Webhook status updated', 'success');
                await this.loadWebhooks();
            } else {
                const result = await response.json();
                throw new Error(result.error || 'Failed to toggle webhook');
            }
        } catch (error) {
            this.showNotification(`Failed to toggle webhook: ${error.message}`, 'error');
        }
    }

    async deleteWebhook(webhookId) {
        if (!confirm('Are you sure you want to delete this webhook?')) return;
        
        try {
            const response = await fetch(`/api/webhooks/${webhookId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                this.showNotification('Webhook deleted successfully', 'success');
                await this.loadWebhooks();
            } else {
                const result = await response.json();
                throw new Error(result.error || 'Failed to delete webhook');
            }
        } catch (error) {
            this.showNotification(`Failed to delete webhook: ${error.message}`, 'error');
        }
    }

    handleJobStatusUpdate(data) {
        if (data.job_id !== this.currentJobId) return;

        console.log('Job status update:', data);
        
        switch (data.status) {
            case 'processing':
                this.updateProgress(data.progress, data.current_step);
                break;
            case 'completed':
                this.handleJobCompleted(data);
                break;
            case 'failed':
                this.handleJobFailed(data);
                break;
        }
    }

    handleProgressUpdate(data) {
        if (data.job_id !== this.currentJobId) return;
        this.updateProgress(data.progress, data.current_step);
    }

    handleStepUpdate(data) {
        if (data.job_id !== this.currentJobId) return;
        this.addToProcessingLog(data.current_step);
    }

    updateProgress(progress, step) {
        const progressBar = document.getElementById('progressBar');
        const progressPercent = document.getElementById('progressPercent');
        const currentStep = document.getElementById('currentStep');

        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
        }

        if (progressPercent) {
            progressPercent.textContent = `${Math.round(progress)}%`;
        }

        if (currentStep && step) {
            currentStep.textContent = step;
        }
    }

    addToProcessingLog(message) {
        const logContainer = document.getElementById('processingLog');
        if (!logContainer) return;

        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry info';
        logEntry.textContent = `[${timestamp}] ${message}`;
        
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    handleJobCompleted(data) {
        this.updateProgress(100, 'Processing completed successfully!');
        this.addToProcessingLog('✅ Processing completed successfully!');
        
        setTimeout(() => {
            this.hideProcessingCard();
            this.showResultsCard();
            this.isProcessing = false;
            this.loadRecentJobs();
        }, 2000);
    }

    handleJobFailed(data) {
        this.addToProcessingLog(`❌ Processing failed: ${data.error_message}`);
        this.showNotification(`Processing failed: ${data.error_message}`, 'error');
        
        setTimeout(() => {
            this.hideProcessingCard();
            this.isProcessing = false;
        }, 3000);
    }

    showProcessingCard() {
        const card = document.getElementById('processingCard');
        if (card) {
            card.style.display = 'block';
            card.classList.add('fade-in');
        }
        
        // Reset progress
        this.updateProgress(0, 'Initializing...');
        
        // Clear previous log
        const logContainer = document.getElementById('processingLog');
        if (logContainer) {
            logContainer.innerHTML = '<div class="text-muted">Processing log will appear here...</div>';
        }
    }

    hideProcessingCard() {
        const card = document.getElementById('processingCard');
        if (card) {
            card.style.display = 'none';
        }
    }

    showResultsCard() {
        const card = document.getElementById('resultsCard');
        if (card) {
            card.style.display = 'block';
            card.classList.add('fade-in');
        }
    }

    hideResultsCard() {
        const card = document.getElementById('resultsCard');
        if (card) {
            card.style.display = 'none';
        }
    }

    async downloadFile(fileType) {
        if (!this.currentJobId) {
            this.showNotification('No job available for download', 'error');
            return;
        }

        try {
            const response = await fetch(`/download/${this.currentJobId}/${fileType}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${fileType}_${this.currentJobId}.${this.getFileExtension(fileType)}`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.showNotification(`${fileType.toUpperCase()} downloaded successfully`, 'success');
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Download failed');
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showNotification(`Download failed: ${error.message}`, 'error');
        }
    }

    getFileExtension(fileType) {
        const extensions = {
            'pdf': 'pdf',
            'transcript': 'txt',
            'audio': 'wav'
        };
        return extensions[fileType] || 'bin';
    }

    async loadRecentJobs() {
        try {
            const response = await fetch('/jobs');
            const data = await response.json();
            
            if (response.ok) {
                this.displayRecentJobs(data.jobs);
            }
        } catch (error) {
            console.error('Failed to load recent jobs:', error);
        }
    }

    displayRecentJobs(jobs) {
        const container = document.getElementById('recentJobs');
        if (!container) return;

        if (jobs.length === 0) {
            container.innerHTML = '<div class="text-muted">No recent jobs</div>';
            const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');
            if (bulkDeleteBtn) bulkDeleteBtn.style.display = 'none';
            return;
        }

        // Show bulk delete button if there are deletable jobs
        const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');
        const hasDeletableJobs = jobs.some(job => 
            job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled'
        );
        if (bulkDeleteBtn) {
            bulkDeleteBtn.style.display = hasDeletableJobs ? 'inline-block' : 'none';
        }

        const jobsHtml = jobs.slice(0, 5).map(job => {
            const canDelete = job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled';
            const statusClass = `job-status ${job.status}`;
            const progress = Math.round(job.progress || 0);
            const duration = job.duration ? `${Math.round(job.duration)}s` : 'N/A';
            const identifier = job.identifier || (job.job_type === 'url' ? 'URL' : 'File');
            const shortIdentifier = identifier.length > 30 ? identifier.substring(0, 30) + '...' : identifier;
            
            return `
                <div class="job-item border rounded p-2 mb-2">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <div class="d-flex align-items-center text-truncate" style="max-width: 200px;" title="${this.escapeHtml(identifier)}">
                            ${canDelete ? `
                                <input type="checkbox" class="form-check-input me-2 job-checkbox" 
                                       value="${job.job_id}" 
                                       onchange="videoProcessor.updateBulkDeleteButton()">
                            ` : ''}
                            <div class="text-truncate">
                                <small class="text-muted">${job.job_type === 'url' ? '🔗' : '📁'}</small>
                                <span class="small">${this.escapeHtml(shortIdentifier)}</span>
                            </div>
                        </div>
                        <span class="badge bg-${this.getStatusBadgeColor(job.status)}">${job.status}</span>
                    </div>
                    ${job.status === 'processing' || job.status === 'pending' ? `
                        <div class="progress mb-2" style="height: 4px;">
                            <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                 style="width: ${progress}%"></div>
                        </div>
                    ` : ''}
                    <div class="d-flex justify-content-between align-items-center">
                        <div class="small text-muted">
                            ${job.current_step || 'No current step'} • ${duration}
                        </div>
                        <div class="btn-group btn-group-sm">
                            ${job.status === 'completed' ? `
                                <a href="/job/${job.job_id}/transcript" class="btn btn-sm btn-outline-primary" title="View">
                                    <i class="fas fa-eye"></i>
                                </a>
                                <button class="btn btn-sm btn-outline-danger" 
                                        onclick="videoProcessor.deleteJob('${job.job_id}')" 
                                        title="Delete">
                                    <i class="fas fa-trash"></i>
                                </button>
                            ` : ''}
                            ${job.status === 'failed' || job.status === 'cancelled' ? `
                                <button class="btn btn-sm btn-outline-warning" 
                                        onclick="videoProcessor.retryJob('${job.job_id}')" 
                                        title="Retry">
                                    <i class="fas fa-redo"></i>
                                </button>
                                <button class="btn btn-sm btn-outline-danger" 
                                        onclick="videoProcessor.deleteJob('${job.job_id}')" 
                                        title="Delete">
                                    <i class="fas fa-trash"></i>
                                </button>
                            ` : ''}
                            ${job.status === 'processing' || job.status === 'pending' ? `
                                <button class="btn btn-sm btn-outline-danger" 
                                        onclick="videoProcessor.cancelJob('${job.job_id}')" 
                                        title="Cancel">
                                    <i class="fas fa-stop"></i>
                                </button>
                            ` : ''}
                        </div>
                    </div>
                    ${job.error_message ? `
                        <div class="small text-danger mt-1">
                            <i class="fas fa-exclamation-triangle me-1"></i>
                            ${this.escapeHtml(job.error_message.substring(0, 100))}
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');

        container.innerHTML = jobsHtml;
        this.updateBulkDeleteButton();
    }

    updateBulkDeleteButton() {
        const checkboxes = document.querySelectorAll('.job-checkbox:checked');
        const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');
        if (bulkDeleteBtn) {
            if (checkboxes.length > 0) {
                bulkDeleteBtn.style.display = 'inline-block';
                bulkDeleteBtn.innerHTML = `<i class="fas fa-trash"></i> Delete Selected (${checkboxes.length})`;
            } else {
                bulkDeleteBtn.innerHTML = '<i class="fas fa-trash"></i> Delete Selected';
            }
        }
    }

    handleBulkDelete() {
        const checkboxes = document.querySelectorAll('.job-checkbox:checked');
        const jobIds = Array.from(checkboxes).map(cb => cb.value);
        this.bulkDeleteJobs(jobIds);
    }

    getStatusBadgeColor(status) {
        const colors = {
            'completed': 'success',
            'processing': 'primary',
            'pending': 'secondary',
            'failed': 'danger',
            'cancelled': 'warning'
        };
        return colors[status] || 'secondary';
    }

    async retryJob(jobId) {
        if (!confirm('Retry this job with the same settings?')) {
            return;
        }

        try {
            const response = await fetch(`/job/${jobId}/retry`, {
                method: 'POST'
            });

            const data = await response.json();

            if (response.ok) {
                this.showNotification(`Job retry started. New job ID: ${data.new_job_id.substring(0, 8)}...`, 'success');
                // Reload recent jobs to show the new job
                setTimeout(() => this.loadRecentJobs(), 1000);
            } else {
                throw new Error(data.error || 'Retry failed');
            }
        } catch (error) {
            this.showNotification(`Failed to retry job: ${error.message}`, 'error');
        }
    }

    async cancelJob(jobId) {
        if (!confirm('Cancel this job?')) {
            return;
        }

        try {
            const response = await fetch(`/job/${jobId}/cancel`, {
                method: 'POST'
            });

            const data = await response.json();

            if (response.ok) {
                this.showNotification('Job cancelled successfully', 'success');
                setTimeout(() => this.loadRecentJobs(), 500);
            } else {
                throw new Error(data.error || 'Cancel failed');
            }
        } catch (error) {
            this.showNotification(`Failed to cancel job: ${error.message}`, 'error');
        }
    }

    async deleteJob(jobId) {
        if (!confirm('Are you sure you want to delete this job? This action cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch(`/job/${jobId}`, {
                method: 'DELETE'
            });

            const data = await response.json();

            if (response.ok) {
                this.showNotification('Job deleted successfully', 'success');
                setTimeout(() => this.loadRecentJobs(), 500);
            } else {
                throw new Error(data.error || 'Delete failed');
            }
        } catch (error) {
            this.showNotification(`Failed to delete job: ${error.message}`, 'error');
        }
    }

    async bulkDeleteJobs(jobIds) {
        if (!jobIds || jobIds.length === 0) {
            this.showNotification('No jobs selected', 'warning');
            return;
        }

        if (!confirm(`Are you sure you want to delete ${jobIds.length} job(s)? This action cannot be undone.`)) {
            return;
        }

        try {
            const response = await fetch('/api/jobs/bulk-delete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ job_ids: jobIds })
            });

            const data = await response.json();

            if (response.ok) {
                this.showNotification(`Successfully deleted ${data.deleted_count} job(s)`, 'success');
                if (data.failed_count > 0) {
                    this.showNotification(`${data.failed_count} job(s) failed to delete`, 'warning');
                }
                setTimeout(() => this.loadRecentJobs(), 500);
            } else {
                throw new Error(data.error || 'Bulk delete failed');
            }
        } catch (error) {
            this.showNotification(`Failed to delete jobs: ${error.message}`, 'error');
        }
    }

    async exportJobs(format) {
        try {
            // Get current filter values from the page if available
            const statusFilter = document.getElementById('filterStatus')?.value || '';
            const typeFilter = document.getElementById('filterType')?.value || '';
            const searchQuery = document.getElementById('searchJobs')?.value || '';
            
            const params = new URLSearchParams();
            params.append('format', format);
            if (statusFilter) params.append('status', statusFilter);
            if (typeFilter) params.append('type', typeFilter);
            if (searchQuery) params.append('search', searchQuery);
            
            const response = await fetch(`/api/jobs/export?${params.toString()}`);
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Export failed');
            }
            
            // Get filename from Content-Disposition header or use default
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = `jobs_export_${new Date().toISOString().split('T')[0]}.${format}`;
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
                if (filenameMatch) {
                    filename = filenameMatch[1];
                }
            }
            
            // Download the file
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            this.showNotification(`Jobs exported successfully as ${format.toUpperCase()}`, 'success');
        } catch (error) {
            this.showNotification(`Failed to export jobs: ${error.message}`, 'error');
        }
    }

    showNotification(message, type = 'info') {
        const toast = document.getElementById('notificationToast');
        const toastMessage = document.getElementById('toastMessage');
        
        if (!toast || !toastMessage) return;

        // Update message and styling
        toastMessage.textContent = message;
        
        // Remove existing type classes
        toast.classList.remove('text-primary', 'text-success', 'text-warning', 'text-danger');
        
        // Add appropriate type class
        const typeClasses = {
            'success': 'text-success',
            'warning': 'text-warning',
            'error': 'text-danger',
            'info': 'text-primary'
        };
        
        if (typeClasses[type]) {
            toast.classList.add(typeClasses[type]);
        }

        // Show toast
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }

    updateFormValidation(activeTab) {
        // Enable/disable streaming checkbox based on active tab
        const streamingCheckbox = document.getElementById('streaming');
        if (streamingCheckbox) {
            const isUrlTab = activeTab.getAttribute('data-bs-target') === '#url-pane';
            streamingCheckbox.disabled = !isUrlTab;
            if (!isUrlTab) {
                streamingCheckbox.checked = false;
            }
        }
    }

    isVideoFile(filename) {
        const videoExtensions = ['.mp4', '.mkv', '.mov', '.webm', '.avi', '.wmv'];
        const audioExtensions = ['.mp3', '.m4a', '.wav', '.flac', '.aac'];
        const allExtensions = [...videoExtensions, ...audioExtensions];
        
        return allExtensions.some(ext => filename.toLowerCase().endsWith(ext));
    }

    async handleSearch(resetPage = false) {
        const queryInput = document.getElementById('searchQuery');
        const targetLanguageSelect = document.getElementById('searchTargetLanguage');
        const searchModeSelect = document.getElementById('searchMode');
        
        const query = queryInput.value.trim();
        if (!query) {
            this.showNotification('Please enter a search query', 'warning');
            return;
        }
        
        // Reset to page 1 if this is a new search (not pagination)
        if (resetPage || !this.lastSearchQuery || this.lastSearchQuery !== query) {
            this.currentSearchPage = 1;
            this.lastSearchQuery = query;
        }

        const targetLanguage = targetLanguageSelect.value || null;
        const searchMode = searchModeSelect ? searchModeSelect.value : 'hybrid';

        // Get hybrid search weights if in hybrid mode
        let semanticWeight = null;
        let keywordWeight = null;
        if (searchMode === 'hybrid') {
            const semanticWeightInput = document.getElementById('semanticWeight');
            const keywordWeightInput = document.getElementById('keywordWeight');
            if (semanticWeightInput && keywordWeightInput) {
                // Convert percentage to decimal (0-1)
                semanticWeight = parseFloat(semanticWeightInput.value) / 100;
                keywordWeight = parseFloat(keywordWeightInput.value) / 100;
            }
        }

        // Get advanced filter values
        const dateFromInput = document.getElementById('filterDateFrom');
        const dateToInput = document.getElementById('filterDateTo');
        const jobTypeInput = document.getElementById('filterJobType');
        const jobStatusInput = document.getElementById('filterJobStatus');
        const originalLanguageInput = document.getElementById('filterOriginalLanguage');

        try {
            // Show loading state
            const searchButton = document.querySelector('#searchForm button[type="submit"]');
            const originalText = searchButton.innerHTML;
            searchButton.disabled = true;
            searchButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Searching...';

            const requestBody = {
                query: query,
                target_language: targetLanguage,
                search_mode: searchMode,
                limit: 500, // Fetch up to 500 results for pagination
                min_score: 0.3,
                page: this.currentSearchPage,
                per_page: this.searchPerPage
            };
            
            // Add weights only if in hybrid mode and weights are provided
            if (searchMode === 'hybrid' && semanticWeight !== null && keywordWeight !== null) {
                requestBody.semantic_weight = semanticWeight;
                requestBody.keyword_weight = keywordWeight;
            }
            
            // Add advanced filters if provided
            if (dateFromInput && dateFromInput.value) {
                requestBody.date_from = dateFromInput.value + 'T00:00:00';
            }
            if (dateToInput && dateToInput.value) {
                requestBody.date_to = dateToInput.value + 'T23:59:59';
            }
            if (jobTypeInput && jobTypeInput.value) {
                requestBody.job_type = jobTypeInput.value;
            }
            if (jobStatusInput && jobStatusInput.value) {
                requestBody.job_status = jobStatusInput.value;
            }
            if (originalLanguageInput && originalLanguageInput.value) {
                requestBody.original_language = originalLanguageInput.value;
            }

            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            const data = await response.json();

            if (response.ok) {
                this.displaySearchResults(data);
                const totalMsg = data.total > 0 
                    ? `Found ${data.total} result${data.total !== 1 ? 's' : ''} (showing page ${data.page} of ${data.total_pages})`
                    : 'No results found';
                this.showNotification(totalMsg, data.total > 0 ? 'success' : 'info');
            } else {
                throw new Error(data.error || 'Search failed');
            }
        } catch (error) {
            console.error('Search error:', error);
            this.showNotification(`Search failed: ${error.message}`, 'error');
            document.getElementById('searchResults').style.display = 'none';
        } finally {
            // Restore button
            const searchButton = document.querySelector('#searchForm button[type="submit"]');
            if (searchButton) {
                searchButton.disabled = false;
                searchButton.innerHTML = '<i class="fas fa-search me-2"></i>Search';
            }
        }
    }

    displaySearchResults(data) {
        const resultsContainer = document.getElementById('searchResults');
        const resultsList = document.getElementById('searchResultsList');

        if (!resultsContainer || !resultsList) return;

        // Store search results for export
        this.lastSearchResults = data;
        
        // Update current page
        if (data.page) {
            this.currentSearchPage = data.page;
        }

        if (data.count === 0 || !data.results || data.results.length === 0) {
            resultsList.innerHTML = '<div class="text-muted p-3">No results found</div>';
            resultsContainer.style.display = 'block';
            // Hide pagination if no results
            const paginationContainer = document.getElementById('searchPagination');
            if (paginationContainer) {
                paginationContainer.style.display = 'none';
            }
            return;
        }

        const resultsHtml = data.results.map((result, index) => {
            const similarityPercent = Math.round(result.similarity * 100);
            const timeStr = this.formatTime(result.start_time);
            
            // Show score breakdown for hybrid mode
            let scoreInfo = `${similarityPercent}% match`;
            if (data.search_mode === 'hybrid' && result.semantic_score !== undefined && result.keyword_score !== undefined) {
                const semanticPercent = Math.round(result.semantic_score * 100);
                const keywordPercent = Math.round(result.keyword_score * 100);
                scoreInfo = `${similarityPercent}% (S:${semanticPercent}% K:${keywordPercent}%)`;
            }
            
            // Show translation indicator if result was translated
            const translationIndicator = result.original_language !== (data.target_language || result.original_language)
                ? '<span class="badge bg-info ms-2" title="Translated from ' + result.original_language + '">Translated</span>'
                : '';

            return `
                <div class="list-group-item" id="result-${result.chunk_id}">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <small class="text-muted">#${index + 1} • ${timeStr}</small>
                        <span class="badge bg-success" title="Combined score${data.search_mode === 'hybrid' ? ' (Semantic + Keyword)' : ''}">${scoreInfo}</span>
                    </div>
                    <p class="mb-2">${this.highlightQuery(result.text, data.query)}</p>
                    <div class="d-flex justify-content-between align-items-center">
                        <small class="text-muted">
                            Job: ${result.job_id.substring(0, 8)}...
                            ${translationIndicator}
                        </small>
                        <div class="btn-group btn-group-sm" role="group">
                            <button class="btn btn-sm btn-outline-info" onclick="videoProcessor.toggleContext('${result.chunk_id}', '${data.target_language || ''}')" id="context-btn-${result.chunk_id}">
                                <i class="fas fa-expand-alt me-1"></i>Context
                            </button>
                            <button class="btn btn-sm btn-outline-primary" onclick="videoProcessor.jumpToTimestamp('${result.job_id}', ${result.start_time})">
                                <i class="fas fa-play me-1"></i>View
                            </button>
                        </div>
                    </div>
                    <div id="context-${result.chunk_id}" class="mt-3" style="display: none;">
                        <div class="text-center">
                            <div class="spinner-border spinner-border-sm text-primary" role="status">
                                <span class="visually-hidden">Loading context...</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        resultsList.innerHTML = resultsHtml;
        resultsContainer.style.display = 'block';
        
        // Add pagination controls
        this.updatePaginationControls(data);
        
        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    updatePaginationControls(data) {
        let paginationContainer = document.getElementById('searchPagination');
        if (!paginationContainer) {
            // Create pagination container if it doesn't exist
            const resultsContainer = document.getElementById('searchResults');
            if (resultsContainer) {
                paginationContainer = document.createElement('div');
                paginationContainer.id = 'searchPagination';
                paginationContainer.className = 'mt-3';
                resultsContainer.appendChild(paginationContainer);
            } else {
                return;
            }
        }
        
        const total = data.total || 0;
        const currentPage = data.page || 1;
        const totalPages = data.total_pages || 0;
        const perPage = data.per_page || 20;
        
        if (totalPages <= 1) {
            paginationContainer.style.display = 'none';
            return;
        }
        
        paginationContainer.style.display = 'block';
        
        // Calculate page range to show (show max 7 page numbers)
        let startPage = Math.max(1, currentPage - 3);
        let endPage = Math.min(totalPages, currentPage + 3);
        
        // Adjust if we're near the start or end
        if (currentPage <= 4) {
            endPage = Math.min(7, totalPages);
        }
        if (currentPage >= totalPages - 3) {
            startPage = Math.max(1, totalPages - 6);
        }
        
        // Build pagination HTML
        let paginationHtml = `
            <div class="d-flex justify-content-between align-items-center">
                <div class="text-muted small">
                    Showing ${(currentPage - 1) * perPage + 1} to ${Math.min(currentPage * perPage, total)} of ${total} results
                </div>
                <nav aria-label="Search results pagination">
                    <ul class="pagination pagination-sm mb-0">
        `;
        
        // Previous button
        paginationHtml += `
            <li class="page-item ${currentPage === 1 ? 'disabled' : ''}">
                <a class="page-link" href="#" onclick="event.preventDefault(); videoProcessor.goToSearchPage(${currentPage - 1}); return false;" ${currentPage === 1 ? 'tabindex="-1" aria-disabled="true"' : ''}>
                    <i class="fas fa-chevron-left"></i> Previous
                </a>
            </li>
        `;
        
        // First page
        if (startPage > 1) {
            paginationHtml += `
                <li class="page-item">
                    <a class="page-link" href="#" onclick="event.preventDefault(); videoProcessor.goToSearchPage(1); return false;">1</a>
                </li>
            `;
            if (startPage > 2) {
                paginationHtml += `<li class="page-item disabled"><span class="page-link">...</span></li>`;
            }
        }
        
        // Page numbers
        for (let i = startPage; i <= endPage; i++) {
            paginationHtml += `
                <li class="page-item ${i === currentPage ? 'active' : ''}">
                    <a class="page-link" href="#" onclick="event.preventDefault(); videoProcessor.goToSearchPage(${i}); return false;">${i}</a>
                </li>
            `;
        }
        
        // Last page
        if (endPage < totalPages) {
            if (endPage < totalPages - 1) {
                paginationHtml += `<li class="page-item disabled"><span class="page-link">...</span></li>`;
            }
            paginationHtml += `
                <li class="page-item">
                    <a class="page-link" href="#" onclick="event.preventDefault(); videoProcessor.goToSearchPage(${totalPages}); return false;">${totalPages}</a>
                </li>
            `;
        }
        
        // Next button
        paginationHtml += `
            <li class="page-item ${currentPage === totalPages ? 'disabled' : ''}">
                <a class="page-link" href="#" onclick="event.preventDefault(); videoProcessor.goToSearchPage(${currentPage + 1}); return false;" ${currentPage === totalPages ? 'tabindex="-1" aria-disabled="true"' : ''}>
                    Next <i class="fas fa-chevron-right"></i>
                </a>
            </li>
        `;
        
        paginationHtml += `
                    </ul>
                </nav>
            </div>
        `;
        
        paginationContainer.innerHTML = paginationHtml;
    }
    
    goToSearchPage(page) {
        if (page < 1) return;
        this.currentSearchPage = page;
        // Re-run search with new page (don't reset page, keep current query)
        this.handleSearch(false);
    }

    highlightQuery(text, query) {
        if (!query) return text;
        const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }

    formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }

    jumpToTimestamp(jobId, timestamp) {
        // Navigate to transcript viewer with timestamp
        const url = `/job/${jobId}/transcript?t=${timestamp}`;
        window.open(url, '_blank');
        this.showNotification(`Opening transcript at ${this.formatTime(timestamp)}...`, 'info');
    }

    async toggleContext(chunkId, targetLanguage) {
        const contextContainer = document.getElementById(`context-${chunkId}`);
        const contextButton = document.getElementById(`context-btn-${chunkId}`);
        
        if (!contextContainer || !contextButton) return;
        
        // If already visible, hide it
        if (contextContainer.style.display !== 'none') {
            contextContainer.style.display = 'none';
            contextButton.innerHTML = '<i class="fas fa-expand-alt me-1"></i>Context';
            return;
        }
        
        // Show loading state
        contextContainer.style.display = 'block';
        contextButton.disabled = true;
        contextButton.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Loading...';
        
        try {
            // Build URL with parameters
            let url = `/api/search/context/${chunkId}?context_before=2&context_after=2`;
            if (targetLanguage) {
                url += `&target_language=${encodeURIComponent(targetLanguage)}`;
            }
            
            const response = await fetch(url);
            const data = await response.json();
            
            if (response.ok && !data.error) {
                this.displayContext(chunkId, data);
                contextButton.innerHTML = '<i class="fas fa-compress-alt me-1"></i>Hide';
            } else {
                throw new Error(data.error || 'Failed to load context');
            }
        } catch (error) {
            console.error('Error loading context:', error);
            contextContainer.innerHTML = `
                <div class="alert alert-warning mb-0">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Failed to load context: ${error.message}
                </div>
            `;
        } finally {
            contextButton.disabled = false;
        }
    }

    displayContext(chunkId, contextData) {
        const contextContainer = document.getElementById(`context-${chunkId}`);
        if (!contextContainer) return;
        
        const query = this.lastSearchResults?.query || '';
        
        // Build context HTML
        let html = '<div class="context-section">';
        
        // Before chunks
        if (contextData.before && contextData.before.length > 0) {
            html += '<div class="context-before mb-2">';
            html += '<small class="text-muted fw-bold"><i class="fas fa-arrow-up me-1"></i>Previous context:</small>';
            contextData.before.forEach((chunk, idx) => {
                html += `
                    <div class="context-chunk p-2 mb-1 bg-light rounded small">
                        <div class="d-flex justify-content-between align-items-start">
                            <span class="text-muted">${this.formatTime(chunk.start_time)}</span>
                            <span class="badge bg-secondary">Chunk ${chunk.chunk_index}</span>
                        </div>
                        <p class="mb-0 mt-1">${this.highlightQuery(chunk.text, query)}</p>
                    </div>
                `;
            });
            html += '</div>';
        }
        
        // Current chunk (highlighted)
        html += '<div class="context-current mb-2">';
        html += '<small class="text-primary fw-bold"><i class="fas fa-bullseye me-1"></i>Matched result:</small>';
        html += `
            <div class="context-chunk p-2 mb-1 bg-primary bg-opacity-10 border border-primary rounded">
                <div class="d-flex justify-content-between align-items-start">
                    <span class="text-primary fw-bold">${this.formatTime(contextData.chunk.start_time)}</span>
                    <span class="badge bg-primary">Chunk ${contextData.chunk.chunk_index}</span>
                </div>
                <p class="mb-0 mt-1 fw-semibold">${this.highlightQuery(contextData.chunk.text, query)}</p>
            </div>
        `;
        html += '</div>';
        
        // After chunks
        if (contextData.after && contextData.after.length > 0) {
            html += '<div class="context-after">';
            html += '<small class="text-muted fw-bold"><i class="fas fa-arrow-down me-1"></i>Following context:</small>';
            contextData.after.forEach((chunk, idx) => {
                html += `
                    <div class="context-chunk p-2 mb-1 bg-light rounded small">
                        <div class="d-flex justify-content-between align-items-start">
                            <span class="text-muted">${this.formatTime(chunk.start_time)}</span>
                            <span class="badge bg-secondary">Chunk ${chunk.chunk_index}</span>
                        </div>
                        <p class="mb-0 mt-1">${this.highlightQuery(chunk.text, query)}</p>
                    </div>
                `;
            });
            html += '</div>';
        }
        
        html += '</div>';
        
        contextContainer.innerHTML = html;
    }

    // Batch Processing Methods
    handleBatchFilesChange(event) {
        const files = Array.from(event.target.files);
        const filesList = document.getElementById('batchFilesList');
        if (!filesList) return;

        if (files.length === 0) {
            filesList.innerHTML = '';
            return;
        }

        if (files.length > 50) {
            this.showNotification('Maximum 50 files allowed. Only first 50 will be processed.', 'warning');
            files = files.slice(0, 50);
        }

        let html = '<div class="list-group">';
        files.forEach((file, index) => {
            const size = (file.size / (1024 * 1024)).toFixed(2);
            html += `
                <div class="list-group-item d-flex justify-content-between align-items-center">
                    <div>
                        <i class="fas fa-file me-2"></i>
                        <strong>${this.escapeHtml(file.name)}</strong>
                        <small class="text-muted ms-2">${size} MB</small>
                    </div>
                    <button class="btn btn-sm btn-outline-danger" onclick="this.closest('.list-group-item').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
        });
        html += '</div>';
        filesList.innerHTML = html;
    }

    async startBatchProcessing() {
        const batchFilesInput = document.getElementById('batchFiles');
        const batchUrlsTextarea = document.getElementById('batchUrls');
        const activeTab = document.querySelector('#batchTabs .nav-link.active');

        let items = [];

        // Check which tab is active
        if (activeTab && activeTab.id === 'batch-files-tab') {
            // File upload mode
            const files = Array.from(batchFilesInput?.files || []);
            if (files.length === 0) {
                this.showNotification('Please select at least one file', 'warning');
                return;
            }

            if (files.length > 50) {
                this.showNotification('Maximum 50 files allowed', 'error');
                return;
            }

            // Upload files first
            const formData = new FormData();
            files.forEach(file => {
                formData.append('files', file);
            });

            try {
                const response = await fetch('/batch/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Upload failed');
                }

                const data = await response.json();
                items = data.files.map(file => ({
                    type: 'file',
                    identifier: file.filename
                }));
            } catch (error) {
                this.showNotification(`Upload failed: ${error.message}`, 'error');
                return;
            }
        } else {
            // URL mode
            const urls = batchUrlsTextarea?.value.split('\n')
                .map(url => url.trim())
                .filter(url => url.length > 0);

            if (urls.length === 0) {
                this.showNotification('Please enter at least one URL', 'warning');
                return;
            }

            if (urls.length > 50) {
                this.showNotification('Maximum 50 URLs allowed', 'error');
                return;
            }

            items = urls.map(url => ({
                type: 'url',
                identifier: url
            }));
        }

        // Get processing options
        const options = {
            language: document.getElementById('batchLanguage')?.value || 'auto',
            whisper_model: document.getElementById('batchWhisperModel')?.value || 'medium',
            beam_size: 5,
            transcribe_only: document.getElementById('batchTranscribeOnly')?.checked || false,
            streaming: false,
            kf_method: 'scene',
            max_fps: 1.0,
            min_scene_diff: 0.45,
            report_style: document.getElementById('batchReportStyle')?.value || 'book'
        };

        // Create batch
        try {
            const response = await fetch('/batch/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    items: items,
                    ...options
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Batch creation failed');
            }

            const data = await response.json();
            this.currentBatchId = data.batch_id;
            this.showNotification(`Batch processing started with ${data.total_jobs} jobs`, 'success');
            
            // Show batch status card
            document.getElementById('batchStatusCard').style.display = 'block';
            
            // Start monitoring batch status
            this.startBatchStatusMonitoring();
        } catch (error) {
            this.showNotification(`Failed to start batch: ${error.message}`, 'error');
        }
    }

    startBatchStatusMonitoring() {
        if (this.batchStatusInterval) {
            clearInterval(this.batchStatusInterval);
        }

        this.batchStatusInterval = setInterval(() => {
            this.updateBatchStatus();
        }, 2000); // Update every 2 seconds

        // Initial update
        this.updateBatchStatus();
    }

    async updateBatchStatus() {
        if (!this.currentBatchId) return;

        try {
            const response = await fetch(`/batch/${this.currentBatchId}`);
            if (!response.ok) {
                throw new Error('Failed to get batch status');
            }

            const batchData = await response.json();
            this.updateBatchStatusUI(batchData);

            // Get individual job statuses
            const jobsResponse = await fetch(`/batch/${this.currentBatchId}/jobs`);
            if (jobsResponse.ok) {
                const jobsData = await jobsResponse.json();
                this.updateBatchJobsList(jobsData.jobs);
            }

            // Stop monitoring if batch is complete
            if (batchData.status === 'completed' || batchData.status === 'failed' || batchData.status === 'cancelled') {
                if (this.batchStatusInterval) {
                    clearInterval(this.batchStatusInterval);
                    this.batchStatusInterval = null;
                }
            }
        } catch (error) {
            console.error('Error updating batch status:', error);
        }
    }

    updateBatchStatusUI(batchData) {
        const total = batchData.total_count || 0;
        const completed = batchData.completed_count || 0;
        const failed = batchData.failed_count || 0;
        const progress = total > 0 ? ((completed + failed) / total) * 100 : 0;

        // Update progress bar
        const progressBar = document.getElementById('batchProgressBar');
        const progressPercent = document.getElementById('batchProgressPercent');
        const progressText = document.getElementById('batchProgressText');

        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.textContent = `${completed + failed} / ${total}`;
        }
        if (progressPercent) {
            progressPercent.textContent = `${Math.round(progress)}%`;
        }
        if (progressText) {
            progressText.textContent = `${completed + failed} / ${total}`;
        }

        // Update counts
        document.getElementById('batchTotalCount').textContent = total;
        document.getElementById('batchCompletedCount').textContent = completed;
        document.getElementById('batchFailedCount').textContent = failed;

        // Update status badge
        const statusBadge = document.getElementById('batchStatusBadge');
        if (statusBadge) {
            statusBadge.textContent = batchData.status.charAt(0).toUpperCase() + batchData.status.slice(1);
            statusBadge.className = 'badge ' + (
                batchData.status === 'completed' ? 'bg-success' :
                batchData.status === 'failed' ? 'bg-danger' :
                batchData.status === 'cancelled' ? 'bg-secondary' :
                'bg-primary'
            );
        }
    }

    updateBatchJobsList(jobs) {
        const jobsList = document.getElementById('batchJobsList');
        if (!jobsList) return;

        if (jobs.length === 0) {
            jobsList.innerHTML = '<div class="text-muted text-center py-3">No jobs in batch</div>';
            return;
        }

        let html = '';
        jobs.forEach((job, index) => {
            const identifier = job.identifier.length > 50 
                ? job.identifier.substring(0, 50) + '...' 
                : job.identifier;
            
            const statusClass = 
                job.status === 'completed' ? 'success' :
                job.status === 'failed' ? 'danger' :
                job.status === 'processing' ? 'primary' :
                'secondary';

            html += `
                <div class="list-group-item">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <div class="d-flex align-items-center mb-1">
                                <span class="badge bg-${statusClass} me-2">${job.status}</span>
                                <small class="text-muted">#${index + 1}</small>
                            </div>
                            <div class="small">${this.escapeHtml(identifier)}</div>
                            ${job.current_step ? `<div class="small text-muted mt-1">${this.escapeHtml(job.current_step)}</div>` : ''}
                            ${job.error_message ? `<div class="small text-danger mt-1">${this.escapeHtml(job.error_message)}</div>` : ''}
                            ${job.status === 'processing' ? `
                                <div class="progress mt-2" style="height: 5px;">
                                    <div class="progress-bar" style="width: ${job.progress || 0}%"></div>
                                </div>
                            ` : ''}
                        </div>
                        <div class="ms-2">
                            ${job.status === 'completed' ? `
                                <a href="/job/${job.job_id}/transcript" class="btn btn-sm btn-outline-primary" title="View">
                                    <i class="fas fa-eye"></i>
                                </a>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `;
        });
        jobsList.innerHTML = html;
    }

    async cancelBatch() {
        if (!this.currentBatchId) return;

        if (!confirm('Are you sure you want to cancel this batch?')) {
            return;
        }

        try {
            const response = await fetch(`/batch/${this.currentBatchId}/cancel`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Failed to cancel batch');
            }

            this.showNotification('Batch cancelled', 'success');
            if (this.batchStatusInterval) {
                clearInterval(this.batchStatusInterval);
                this.batchStatusInterval = null;
            }
            this.updateBatchStatus();
        } catch (error) {
            this.showNotification(`Failed to cancel batch: ${error.message}`, 'error');
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Batch Processing Methods
    handleBatchFilesChange(event) {
        const files = Array.from(event.target.files);
        const filesList = document.getElementById('batchFilesList');
        if (!filesList) return;

        if (files.length === 0) {
            filesList.innerHTML = '';
            return;
        }

        let filesToProcess = files;
        if (files.length > 50) {
            this.showNotification('Maximum 50 files allowed. Only first 50 will be processed.', 'warning');
            filesToProcess = files.slice(0, 50);
        }

        let html = '<div class="list-group">';
        filesToProcess.forEach((file, index) => {
            const size = (file.size / (1024 * 1024)).toFixed(2);
            html += `
                <div class="list-group-item d-flex justify-content-between align-items-center">
                    <div>
                        <i class="fas fa-file me-2"></i>
                        <strong>${this.escapeHtml(file.name)}</strong>
                        <small class="text-muted ms-2">${size} MB</small>
                    </div>
                    <button class="btn btn-sm btn-outline-danger" onclick="this.closest('.list-group-item').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
        });
        html += '</div>';
        filesList.innerHTML = html;
    }

    async startBatchProcessing() {
        const batchFilesInput = document.getElementById('batchFiles');
        const batchUrlsTextarea = document.getElementById('batchUrls');
        const activeTab = document.querySelector('#batchTabs .nav-link.active');

        let items = [];

        // Check which tab is active
        if (activeTab && activeTab.id === 'batch-files-tab') {
            // File upload mode
            const files = Array.from(batchFilesInput?.files || []);
            if (files.length === 0) {
                this.showNotification('Please select at least one file', 'warning');
                return;
            }

            if (files.length > 50) {
                this.showNotification('Maximum 50 files allowed', 'error');
                return;
            }

            // Upload files first
            const formData = new FormData();
            files.forEach(file => {
                formData.append('files', file);
            });

            try {
                const response = await fetch('/batch/upload', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Upload failed');
                }

                const data = await response.json();
                items = data.files.map(file => ({
                    type: 'file',
                    identifier: file.filename
                }));
            } catch (error) {
                this.showNotification(`Upload failed: ${error.message}`, 'error');
                return;
            }
        } else {
            // URL mode
            const urls = batchUrlsTextarea?.value.split('\n')
                .map(url => url.trim())
                .filter(url => url.length > 0);

            if (urls.length === 0) {
                this.showNotification('Please enter at least one URL', 'warning');
                return;
            }

            if (urls.length > 50) {
                this.showNotification('Maximum 50 URLs allowed', 'error');
                return;
            }

            items = urls.map(url => ({
                type: 'url',
                identifier: url
            }));
        }

        // Get processing options
        const options = {
            language: document.getElementById('batchLanguage')?.value || 'auto',
            whisper_model: document.getElementById('batchWhisperModel')?.value || 'medium',
            beam_size: 5,
            transcribe_only: document.getElementById('batchTranscribeOnly')?.checked || false,
            streaming: false,
            kf_method: 'scene',
            max_fps: 1.0,
            min_scene_diff: 0.45,
            report_style: document.getElementById('batchReportStyle')?.value || 'book'
        };

        // Create batch
        try {
            const response = await fetch('/batch/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    items: items,
                    ...options
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Batch creation failed');
            }

            const data = await response.json();
            this.currentBatchId = data.batch_id;
            this.showNotification(`Batch processing started with ${data.total_jobs} jobs`, 'success');
            
            // Show batch status card
            document.getElementById('batchStatusCard').style.display = 'block';
            
            // Start monitoring batch status
            this.startBatchStatusMonitoring();
        } catch (error) {
            this.showNotification(`Failed to start batch: ${error.message}`, 'error');
        }
    }

    startBatchStatusMonitoring() {
        if (this.batchStatusInterval) {
            clearInterval(this.batchStatusInterval);
        }

        this.batchStatusInterval = setInterval(() => {
            this.updateBatchStatus();
        }, 2000); // Update every 2 seconds

        // Initial update
        this.updateBatchStatus();
    }

    async updateBatchStatus() {
        if (!this.currentBatchId) return;

        try {
            const response = await fetch(`/batch/${this.currentBatchId}`);
            if (!response.ok) {
                throw new Error('Failed to get batch status');
            }

            const batchData = await response.json();
            this.updateBatchStatusUI(batchData);

            // Get individual job statuses
            const jobsResponse = await fetch(`/batch/${this.currentBatchId}/jobs`);
            if (jobsResponse.ok) {
                const jobsData = await jobsResponse.json();
                this.updateBatchJobsList(jobsData.jobs);
            }

            // Stop monitoring if batch is complete
            if (batchData.status === 'completed' || batchData.status === 'failed' || batchData.status === 'cancelled') {
                if (this.batchStatusInterval) {
                    clearInterval(this.batchStatusInterval);
                    this.batchStatusInterval = null;
                }
            }
        } catch (error) {
            console.error('Error updating batch status:', error);
        }
    }

    updateBatchStatusUI(batchData) {
        const total = batchData.total_count || 0;
        const completed = batchData.completed_count || 0;
        const failed = batchData.failed_count || 0;
        const progress = total > 0 ? ((completed + failed) / total) * 100 : 0;

        // Update progress bar
        const progressBar = document.getElementById('batchProgressBar');
        const progressPercent = document.getElementById('batchProgressPercent');
        const progressText = document.getElementById('batchProgressText');

        if (progressBar) {
            progressBar.style.width = `${progress}%`;
            progressBar.textContent = `${completed + failed} / ${total}`;
        }
        if (progressPercent) {
            progressPercent.textContent = `${Math.round(progress)}%`;
        }
        if (progressText) {
            progressText.textContent = `${completed + failed} / ${total}`;
        }

        // Update counts
        const totalEl = document.getElementById('batchTotalCount');
        const completedEl = document.getElementById('batchCompletedCount');
        const failedEl = document.getElementById('batchFailedCount');
        if (totalEl) totalEl.textContent = total;
        if (completedEl) completedEl.textContent = completed;
        if (failedEl) failedEl.textContent = failed;

        // Update status badge
        const statusBadge = document.getElementById('batchStatusBadge');
        if (statusBadge) {
            statusBadge.textContent = batchData.status.charAt(0).toUpperCase() + batchData.status.slice(1);
            statusBadge.className = 'badge ' + (
                batchData.status === 'completed' ? 'bg-success' :
                batchData.status === 'failed' ? 'bg-danger' :
                batchData.status === 'cancelled' ? 'bg-secondary' :
                'bg-primary'
            );
        }
    }

    updateBatchJobsList(jobs) {
        const jobsList = document.getElementById('batchJobsList');
        if (!jobsList) return;

        if (jobs.length === 0) {
            jobsList.innerHTML = '<div class="text-muted text-center py-3">No jobs in batch</div>';
            return;
        }

        let html = '';
        jobs.forEach((job, index) => {
            const identifier = job.identifier.length > 50 
                ? job.identifier.substring(0, 50) + '...' 
                : job.identifier;
            
            const statusClass = 
                job.status === 'completed' ? 'success' :
                job.status === 'failed' ? 'danger' :
                job.status === 'processing' ? 'primary' :
                'secondary';

            html += `
                <div class="list-group-item">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <div class="d-flex align-items-center mb-1">
                                <span class="badge bg-${statusClass} me-2">${job.status}</span>
                                <small class="text-muted">#${index + 1}</small>
                            </div>
                            <div class="small">${this.escapeHtml(identifier)}</div>
                            ${job.current_step ? `<div class="small text-muted mt-1">${this.escapeHtml(job.current_step)}</div>` : ''}
                            ${job.error_message ? `<div class="small text-danger mt-1">${this.escapeHtml(job.error_message)}</div>` : ''}
                            ${job.status === 'processing' ? `
                                <div class="progress mt-2" style="height: 5px;">
                                    <div class="progress-bar" style="width: ${job.progress || 0}%"></div>
                                </div>
                            ` : ''}
                        </div>
                        <div class="ms-2">
                            ${job.status === 'completed' ? `
                                <a href="/job/${job.job_id}/transcript" class="btn btn-sm btn-outline-primary" title="View">
                                    <i class="fas fa-eye"></i>
                                </a>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `;
        });
        jobsList.innerHTML = html;
    }

    async cancelBatch() {
        if (!this.currentBatchId) return;

        if (!confirm('Are you sure you want to cancel this batch?')) {
            return;
        }

        try {
            const response = await fetch(`/batch/${this.currentBatchId}/cancel`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error('Failed to cancel batch');
            }

            this.showNotification('Batch cancelled', 'success');
            if (this.batchStatusInterval) {
                clearInterval(this.batchStatusInterval);
                this.batchStatusInterval = null;
            }
            this.updateBatchStatus();
        } catch (error) {
            this.showNotification(`Failed to cancel batch: ${error.message}`, 'error');
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    exportSearchResults(format) {
        // Get current search results from the displayed data
        const resultsContainer = document.getElementById('searchResults');
        if (!resultsContainer || resultsContainer.style.display === 'none') {
            this.showNotification('No search results to export', 'warning');
            return;
        }

        // Get the search results data from the last search
        if (!this.lastSearchResults) {
            this.showNotification('No search results available to export', 'warning');
            return;
        }

        const data = this.lastSearchResults;
        const results = data.results || [];

        if (results.length === 0) {
            this.showNotification('No results to export', 'warning');
            return;
        }

        try {
            if (format === 'csv') {
                this.exportToCsv(results, data);
            } else if (format === 'json') {
                this.exportToJson(results, data);
            }
        } catch (error) {
            console.error('Export error:', error);
            this.showNotification(`Export failed: ${error.message}`, 'error');
        }
    }

    exportToCsv(results, searchData) {
        // Create CSV header
        const headers = [
            'Rank',
            'Job ID',
            'Text',
            'Original Text',
            'Start Time',
            'End Time',
            'Duration (s)',
            'Similarity Score',
            'Semantic Score',
            'Keyword Score',
            'Original Language',
            'Target Language',
            'Search Mode',
            'Query'
        ];

        // Create CSV rows
        const rows = results.map((result, index) => {
            const duration = (result.end_time - result.start_time).toFixed(2);
            return [
                index + 1,
                result.job_id,
                this.escapeCsvField(result.text || ''),
                this.escapeCsvField(result.original_text || result.text || ''),
                this.formatTime(result.start_time),
                this.formatTime(result.end_time),
                duration,
                (result.similarity * 100).toFixed(2) + '%',
                (result.semantic_score || 0) * 100,
                (result.keyword_score || 0) * 100,
                result.original_language || 'unknown',
                searchData.target_language || 'original',
                searchData.search_mode || 'semantic',
                searchData.query || ''
            ];
        });

        // Combine header and rows
        const csvContent = [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');

        // Add BOM for Excel compatibility
        const BOM = '\uFEFF';
        const blob = new Blob([BOM + csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `search_results_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);

        this.showNotification(`Exported ${results.length} results to CSV`, 'success');
    }

    exportToJson(results, searchData) {
        // Create export object with metadata
        const exportData = {
            export_info: {
                exported_at: new Date().toISOString(),
                query: searchData.query,
                search_mode: searchData.search_mode,
                target_language: searchData.target_language,
                filters: {
                    date_from: searchData.date_from || null,
                    date_to: searchData.date_to || null,
                    job_type: searchData.job_type || null,
                    job_status: searchData.job_status || null,
                    original_language: searchData.original_language || null
                },
                total_results: results.length
            },
            results: results.map((result, index) => ({
                rank: index + 1,
                job_id: result.job_id,
                chunk_id: result.chunk_id,
                text: result.text,
                original_text: result.original_text || result.text,
                start_time: result.start_time,
                end_time: result.end_time,
                duration: result.end_time - result.start_time,
                similarity_score: result.similarity,
                semantic_score: result.semantic_score || 0,
                keyword_score: result.keyword_score || 0,
                original_language: result.original_language,
                chunk_index: result.chunk_index,
                metadata: result.metadata || {}
            }))
        };

        const jsonContent = JSON.stringify(exportData, null, 2);
        const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `search_results_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);

        this.showNotification(`Exported ${results.length} results to JSON`, 'success');
    }

    escapeCsvField(field) {
        if (field === null || field === undefined) {
            return '';
        }
        const str = String(field);
        // If field contains comma, newline, or quote, wrap in quotes and escape quotes
        if (str.includes(',') || str.includes('\n') || str.includes('"')) {
            return '"' + str.replace(/"/g, '""') + '"';
        }
        return str;
    }
}

// Global functions for HTML onclick handlers
function processNewVideo() {
    const processor = window.videoProcessor;
    if (processor) {
        processor.hideResultsCard();
        processor.currentJobId = null;
        
        // Reset forms
        document.getElementById('fileUploadForm')?.reset();
        document.getElementById('urlForm')?.reset();
        
        // Switch to first tab
        const firstTab = document.getElementById('file-tab');
        if (firstTab) {
            firstTab.click();
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.videoProcessor = new VideoProcessor();
});

// Handle page visibility changes to reconnect socket if needed
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.videoProcessor && !window.videoProcessor.socket.connected) {
        console.log('Reconnecting socket...');
        window.videoProcessor.socket.connect();
    }
});
