/**
 * Jobs Dashboard JavaScript
 * Handles job listing, filtering, sorting, and management
 */

const jobsDashboard = {
    jobs: [],
    filteredJobs: [],
    autoRefreshInterval: null,
    currentPage: 1,
    itemsPerPage: 12,

    init() {
        this.setupEventListeners();
        this.loadJobs();
        this.startAutoRefresh();
    },

    setupEventListeners() {
        // Filter changes
        document.getElementById('filterStatus')?.addEventListener('change', () => this.applyFilters());
        document.getElementById('filterType')?.addEventListener('change', () => this.applyFilters());
        document.getElementById('filterSort')?.addEventListener('change', () => this.applyFilters());
        let searchTimeout;
        document.getElementById('searchJobs')?.addEventListener('input', () => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => this.applyFilters(), 300);
        });

        // Auto-refresh toggle
        document.getElementById('autoRefresh')?.addEventListener('change', (e) => {
            if (e.target.checked) {
                this.startAutoRefresh();
            } else {
                this.stopAutoRefresh();
            }
        });
    },

    async loadJobs() {
        try {
            const status = document.getElementById('filterStatus')?.value || '';
            const type = document.getElementById('filterType')?.value || '';
            const search = document.getElementById('searchJobs')?.value || '';
            const sort = document.getElementById('filterSort')?.value || 'created_desc';

            const params = new URLSearchParams();
            if (status) params.append('status', status);
            if (type) params.append('type', type);
            if (search) params.append('search', search);
            if (sort) params.append('sort', sort);

            const response = await fetch(`/jobs?${params.toString()}`);
            if (!response.ok) {
                throw new Error('Failed to load jobs');
            }

            const data = await response.json();
            this.jobs = data.jobs || [];
            this.filteredJobs = this.jobs;
            
            this.updateStats();
            this.displayJobs();
        } catch (error) {
            console.error('Error loading jobs:', error);
            this.showError('Failed to load jobs');
        }
    },

    applyFilters() {
        this.currentPage = 1;
        this.loadJobs();
    },

    clearFilters() {
        document.getElementById('filterStatus').value = '';
        document.getElementById('filterType').value = '';
        document.getElementById('filterSort').value = 'created_desc';
        document.getElementById('searchJobs').value = '';
        this.applyFilters();
    },

    updateStats() {
        const stats = {
            total: this.jobs.length,
            completed: this.jobs.filter(j => j.status === 'completed').length,
            processing: this.jobs.filter(j => j.status === 'processing' || j.status === 'pending').length,
            failed: this.jobs.filter(j => j.status === 'failed').length
        };

        document.getElementById('statTotal').textContent = stats.total;
        document.getElementById('statCompleted').textContent = stats.completed;
        document.getElementById('statProcessing').textContent = stats.processing;
        document.getElementById('statFailed').textContent = stats.failed;
        document.getElementById('jobsCount').textContent = `${stats.total} job${stats.total !== 1 ? 's' : ''}`;
    },

    displayJobs() {
        const container = document.getElementById('jobsList');
        if (!container) return;

        if (this.filteredJobs.length === 0) {
            container.innerHTML = `
                <div class="col-12 text-center text-muted py-5">
                    <i class="fas fa-inbox fa-3x mb-3"></i>
                    <p>No jobs found</p>
                </div>
            `;
            document.getElementById('jobsPagination').innerHTML = '';
            return;
        }

        // Pagination
        const totalPages = Math.ceil(this.filteredJobs.length / this.itemsPerPage);
        const startIdx = (this.currentPage - 1) * this.itemsPerPage;
        const endIdx = startIdx + this.itemsPerPage;
        const pageJobs = this.filteredJobs.slice(startIdx, endIdx);

        let html = '';
        pageJobs.forEach(job => {
            html += this.renderJobCard(job);
        });

        container.innerHTML = html;
        this.renderPagination(totalPages);
    },

    renderJobCard(job) {
        const statusClass = this.getStatusClass(job.status);
        const typeIcon = job.type === 'batch' ? 'fa-layer-group' : (job.job_type === 'url' ? 'fa-link' : 'fa-file');
        const identifier = this.truncateText(job.identifier, 60);
        const createdDate = new Date(job.created_at).toLocaleString();
        const duration = job.duration ? this.formatDuration(job.duration) : null;

        return `
            <div class="col-md-6 col-lg-4">
                <div class="card job-card h-100" onclick="jobsDashboard.showJobDetails('${job.id}', '${job.type}')">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <span class="badge ${statusClass} status-badge">${job.status}</span>
                            <i class="fas ${typeIcon} text-muted"></i>
                        </div>
                        <h6 class="card-title mb-2">${this.escapeHtml(identifier)}</h6>
                        <div class="small text-muted mb-2">
                            <div><i class="fas fa-calendar me-1"></i>${createdDate}</div>
                            ${duration ? `<div><i class="fas fa-clock me-1"></i>${duration}</div>` : ''}
                        </div>
                        ${job.current_step ? `
                            <div class="small text-muted mb-2">
                                <i class="fas fa-cog me-1"></i>${this.escapeHtml(job.current_step)}
                            </div>
                        ` : ''}
                        ${job.status === 'processing' || job.status === 'pending' ? `
                            <div class="progress progress-thin mb-2">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                     style="width: ${job.progress || 0}%"></div>
                            </div>
                        ` : ''}
                        ${job.type === 'batch' ? `
                            <div class="small">
                                <span class="badge bg-success">${job.completed_count || 0} completed</span>
                                ${job.failed_count > 0 ? `<span class="badge bg-danger">${job.failed_count} failed</span>` : ''}
                                <span class="badge bg-secondary">${job.total_count || 0} total</span>
                            </div>
                        ` : ''}
                        ${job.error_message ? `
                            <div class="small text-danger mt-2">
                                <i class="fas fa-exclamation-triangle me-1"></i>
                                ${this.escapeHtml(job.error_message.substring(0, 100))}
                            </div>
                        ` : ''}
                    </div>
                    <div class="card-footer bg-transparent border-top-0">
                        <div class="btn-group btn-group-sm w-100" role="group" onclick="event.stopPropagation()">
                            ${job.status === 'completed' ? `
                                <a href="/job/${job.job_id}/transcript" class="btn btn-outline-primary btn-sm">
                                    <i class="fas fa-eye me-1"></i>View
                                </a>
                            ` : ''}
                            ${job.status === 'processing' || job.status === 'pending' ? `
                                <button class="btn btn-outline-danger btn-sm" onclick="jobsDashboard.cancelJob('${job.id}', '${job.type}')">
                                    <i class="fas fa-stop me-1"></i>Cancel
                                </button>
                            ` : ''}
                            <button class="btn btn-outline-info btn-sm" onclick="jobsDashboard.showJobDetails('${job.id}', '${job.type}')">
                                <i class="fas fa-info-circle me-1"></i>Details
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    },

    renderPagination(totalPages) {
        const container = document.getElementById('jobsPagination');
        if (!container || totalPages <= 1) {
            container.innerHTML = '';
            return;
        }

        let html = '<nav><ul class="pagination justify-content-center">';
        
        // Previous
        html += `
            <li class="page-item ${this.currentPage === 1 ? 'disabled' : ''}">
                <a class="page-link" href="#" onclick="event.preventDefault(); jobsDashboard.goToPage(${this.currentPage - 1})">
                    <i class="fas fa-chevron-left"></i>
                </a>
            </li>
        `;

        // Page numbers
        for (let i = 1; i <= totalPages; i++) {
            if (i === 1 || i === totalPages || (i >= this.currentPage - 2 && i <= this.currentPage + 2)) {
                html += `
                    <li class="page-item ${i === this.currentPage ? 'active' : ''}">
                        <a class="page-link" href="#" onclick="event.preventDefault(); jobsDashboard.goToPage(${i})">${i}</a>
                    </li>
                `;
            } else if (i === this.currentPage - 3 || i === this.currentPage + 3) {
                html += '<li class="page-item disabled"><span class="page-link">...</span></li>';
            }
        }

        // Next
        html += `
            <li class="page-item ${this.currentPage === totalPages ? 'disabled' : ''}">
                <a class="page-link" href="#" onclick="event.preventDefault(); jobsDashboard.goToPage(${this.currentPage + 1})">
                    <i class="fas fa-chevron-right"></i>
                </a>
            </li>
        `;

        html += '</ul></nav>';
        container.innerHTML = html;
    },

    goToPage(page) {
        const totalPages = Math.ceil(this.filteredJobs.length / this.itemsPerPage);
        if (page >= 1 && page <= totalPages) {
            this.currentPage = page;
            this.displayJobs();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    },

    async showJobDetails(jobId, type) {
        const modal = new bootstrap.Modal(document.getElementById('jobDetailsModal'));
        const content = document.getElementById('jobDetailsContent');
        
        content.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"></div></div>';
        modal.show();

        try {
            if (type === 'batch') {
                const response = await fetch(`/batch/${jobId}`);
                if (!response.ok) throw new Error('Failed to load batch details');
                const batchData = await response.json();
                
                const jobsResponse = await fetch(`/batch/${jobId}/jobs`);
                const jobsData = jobsResponse.ok ? await jobsResponse.json() : { jobs: [] };
                
                content.innerHTML = this.renderBatchDetails(batchData, jobsData.jobs);
            } else {
                const response = await fetch(`/job/${jobId}`);
                if (!response.ok) throw new Error('Failed to load job details');
                const jobData = await response.json();
                content.innerHTML = this.renderJobDetails(jobData);
            }
        } catch (error) {
            content.innerHTML = `<div class="alert alert-danger">Error loading details: ${error.message}</div>`;
        }
    },

    renderJobDetails(job) {
        return `
            <div class="row">
                <div class="col-md-6">
                    <h6>Job Information</h6>
                    <table class="table table-sm">
                        <tr><td><strong>Job ID:</strong></td><td><code>${job.job_id}</code></td></tr>
                        <tr><td><strong>Type:</strong></td><td>${job.job_type}</td></tr>
                        <tr><td><strong>Status:</strong></td><td><span class="badge ${this.getStatusClass(job.status)}">${job.status}</span></td></tr>
                        <tr><td><strong>Progress:</strong></td><td>${job.progress || 0}%</td></tr>
                        <tr><td><strong>Created:</strong></td><td>${new Date(job.created_at).toLocaleString()}</td></tr>
                        ${job.start_time ? `<tr><td><strong>Started:</strong></td><td>${new Date(job.start_time * 1000).toLocaleString()}</td></tr>` : ''}
                        ${job.end_time ? `<tr><td><strong>Ended:</strong></td><td>${new Date(job.end_time * 1000).toLocaleString()}</td></tr>` : ''}
                        ${job.duration ? `<tr><td><strong>Duration:</strong></td><td>${this.formatDuration(job.duration)}</td></tr>` : ''}
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Details</h6>
                    <table class="table table-sm">
                        <tr><td><strong>Identifier:</strong></td><td>${this.escapeHtml(job.identifier)}</td></tr>
                        ${job.current_step ? `<tr><td><strong>Current Step:</strong></td><td>${this.escapeHtml(job.current_step)}</td></tr>` : ''}
                        ${job.error_message ? `<tr><td><strong>Error:</strong></td><td class="text-danger">${this.escapeHtml(job.error_message)}</td></tr>` : ''}
                    </table>
                    ${job.status === 'completed' ? `
                        <div class="mt-3">
                            <a href="/job/${job.job_id}/transcript" class="btn btn-primary">
                                <i class="fas fa-eye me-1"></i>View Transcript
                            </a>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    },

    renderBatchDetails(batch, jobs) {
        return `
            <div class="row">
                <div class="col-md-6">
                    <h6>Batch Information</h6>
                    <table class="table table-sm">
                        <tr><td><strong>Batch ID:</strong></td><td><code>${batch.batch_id}</code></td></tr>
                        <tr><td><strong>Status:</strong></td><td><span class="badge ${this.getStatusClass(batch.status)}">${batch.status}</span></td></tr>
                        <tr><td><strong>Total Jobs:</strong></td><td>${batch.total_count}</td></tr>
                        <tr><td><strong>Completed:</strong></td><td class="text-success">${batch.completed_count}</td></tr>
                        <tr><td><strong>Failed:</strong></td><td class="text-danger">${batch.failed_count}</td></tr>
                        <tr><td><strong>Progress:</strong></td><td>${((batch.completed_count + batch.failed_count) / batch.total_count * 100).toFixed(1)}%</td></tr>
                        <tr><td><strong>Created:</strong></td><td>${new Date(batch.created_at).toLocaleString()}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>Processing Options</h6>
                    <pre class="bg-light p-2 small">${JSON.stringify(batch.options, null, 2)}</pre>
                </div>
            </div>
            <div class="mt-3">
                <h6>Jobs in Batch (${jobs.length})</h6>
                <div class="list-group" style="max-height: 300px; overflow-y: auto;">
                    ${jobs.map((job, idx) => `
                        <div class="list-group-item">
                            <div class="d-flex justify-content-between">
                                <div>
                                    <span class="badge ${this.getStatusClass(job.status)} me-2">${job.status}</span>
                                    <small>#${idx + 1}: ${this.escapeHtml(this.truncateText(job.identifier, 50))}</small>
                                </div>
                                ${job.status === 'completed' ? `
                                    <a href="/job/${job.job_id}/transcript" class="btn btn-sm btn-outline-primary">
                                        <i class="fas fa-eye"></i>
                                    </a>
                                ` : ''}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    },

    async cancelJob(jobId, type) {
        if (!confirm('Are you sure you want to cancel this job?')) {
            return;
        }

        try {
            const url = type === 'batch' ? `/batch/${jobId}/cancel` : `/job/${jobId}/cancel`;
            const response = await fetch(url, { method: 'POST' });
            
            if (!response.ok) {
                throw new Error('Failed to cancel job');
            }

            this.showSuccess('Job cancelled');
            this.loadJobs();
        } catch (error) {
            this.showError(`Failed to cancel job: ${error.message}`);
        }
    },

    refreshJobs() {
        this.loadJobs();
    },

    startAutoRefresh() {
        this.stopAutoRefresh();
        this.autoRefreshInterval = setInterval(() => {
            this.loadJobs();
        }, 5000);
    },

    stopAutoRefresh() {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
        }
    },

    exportJobs() {
        const data = this.filteredJobs.map(job => ({
            id: job.job_id,
            type: job.type === 'batch' ? 'batch' : job.job_type,
            status: job.status,
            identifier: job.identifier,
            created_at: job.created_at,
            progress: job.progress,
            duration: job.duration
        }));

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `jobs_export_${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    },

    getStatusClass(status) {
        const classes = {
            'completed': 'bg-success',
            'processing': 'bg-primary',
            'pending': 'bg-secondary',
            'failed': 'bg-danger',
            'cancelled': 'bg-warning'
        };
        return classes[status] || 'bg-secondary';
    },

    formatDuration(seconds) {
        if (!seconds) return 'N/A';
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    },

    truncateText(text, maxLength) {
        if (!text) return '';
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },


    showSuccess(message) {
        // Simple notification - could be enhanced
        alert(message);
    },

    showError(message) {
        alert(`Error: ${message}`);
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    jobsDashboard.init();
});

