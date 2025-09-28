/**
 * Video Documentation Builder - Frontend JavaScript
 * Handles form submissions, WebSocket communication, and real-time updates
 */

class VideoProcessor {
    constructor() {
        this.socket = null;
        this.currentJobId = null;
        this.isProcessing = false;
        this.init();
    }

    init() {
        this.setupSocketConnection();
        this.setupEventListeners();
        this.loadRecentJobs();
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
                throw new Error(result.error || 'Upload failed');
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
                throw new Error(result.error || 'Processing failed');
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
            return;
        }

        const jobsHtml = jobs.slice(0, 5).map(job => {
            const statusClass = `job-status ${job.status}`;
            const progress = Math.round(job.progress || 0);
            const duration = job.duration ? `${Math.round(job.duration)}s` : 'N/A';
            
            return `
                <div class="job-item">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <div class="text-truncate" style="max-width: 150px;">
                            ${job.job_type === 'url' ? 'URL' : 'File'}
                        </div>
                        <span class="${statusClass}">${job.status}</span>
                    </div>
                    <div class="job-progress mb-2">
                        <div class="job-progress-bar" style="width: ${progress}%"></div>
                    </div>
                    <div class="small text-muted">
                        ${job.current_step || 'No current step'} • ${duration}
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = jobsHtml;
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
