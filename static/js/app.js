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
                this.handleSearch();
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

    async handleSearch() {
        const queryInput = document.getElementById('searchQuery');
        const targetLanguageSelect = document.getElementById('searchTargetLanguage');
        const searchModeSelect = document.getElementById('searchMode');
        
        const query = queryInput.value.trim();
        if (!query) {
            this.showNotification('Please enter a search query', 'warning');
            return;
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
                limit: 20,
                min_score: 0.3
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
                this.showNotification(`Found ${data.count} results`, 'success');
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

        if (data.count === 0) {
            resultsList.innerHTML = '<div class="text-muted p-3">No results found</div>';
            resultsContainer.style.display = 'block';
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
                <div class="list-group-item">
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
                        <button class="btn btn-sm btn-outline-primary" onclick="videoProcessor.jumpToTimestamp('${result.job_id}', ${result.start_time})">
                            <i class="fas fa-play me-1"></i>View
                        </button>
                    </div>
                </div>
            `;
        }).join('');

        resultsList.innerHTML = resultsHtml;
        resultsContainer.style.display = 'block';
        
        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
