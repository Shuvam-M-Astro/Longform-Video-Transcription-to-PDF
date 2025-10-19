# API Documentation

This document provides comprehensive API documentation for the Video Processing System.

## Overview

The Video Processing API provides endpoints for:
- **Authentication**: User login, logout, registration, and API key management
- **Job Management**: Video processing, status tracking, and file downloads
- **Health Monitoring**: System health checks and metrics
- **User Management**: Account management and permissions

## Base URL

- **Development**: `http://localhost:5000`
- **Production**: `https://api.example.com`

## Authentication

The API supports three authentication methods:

### 1. Session Authentication
Login via `/auth/login` and use the session cookie for subsequent requests.

### 2. Bearer Token Authentication
Include the session token in the Authorization header:
```
Authorization: Bearer <session_token>
```

### 3. API Key Authentication
Include the API key in the X-API-Key header:
```
X-API-Key: <api_key>
```

## Interactive Documentation

- **API Documentation**: `/api/docs` - Interactive documentation interface
- **OpenAPI Specification**: `/api/openapi.json` - Machine-readable API spec
- **Endpoints List**: `/api/endpoints` - JSON list of all endpoints

## Authentication Endpoints

### POST /auth/login
Authenticate user and create session.

**Request Body:**
```json
{
  "username": "admin",
  "password": "admin123"
}
```

**Response (200):**
```json
{
  "message": "Login successful",
  "user": {
    "id": "user123",
    "username": "admin",
    "role": "admin",
    "permissions": ["create_job", "view_job", "manage_users"]
  }
}
```

**Response (401):**
```json
{
  "error": "Invalid username or password"
}
```

### POST /auth/logout
Logout user and invalidate session.

**Auth Required:** Yes

**Response (200):**
```json
{
  "message": "Logout successful"
}
```

### GET /auth/me
Get current user information.

**Auth Required:** Yes

**Response (200):**
```json
{
  "user": {
    "id": "user123",
    "username": "admin",
    "role": "admin",
    "permissions": ["create_job", "view_job", "manage_users"]
  }
}
```

### POST /auth/register
Register new user account.

**Request Body:**
```json
{
  "username": "newuser",
  "email": "user@example.com",
  "password": "password123"
}
```

**Response (201):**
```json
{
  "message": "User created successfully",
  "user": {
    "id": "user123",
    "username": "newuser",
    "email": "user@example.com",
    "role": "user"
  }
}
```

### POST /auth/api-keys
Create new API key for programmatic access.

**Auth Required:** Yes (manage_users permission)

**Request Body:**
```json
{
  "key_name": "My API Key",
  "permissions": ["create_job", "view_job"],
  "expires_days": 30
}
```

**Response (201):**
```json
{
  "message": "API key created successfully",
  "api_key": "sk-1234567890abcdef",
  "key_name": "My API Key"
}
```

### GET /auth/api-keys
List user's API keys.

**Auth Required:** Yes (manage_users permission)

**Response (200):**
```json
{
  "api_keys": [
    {
      "id": "key123",
      "key_name": "My API Key",
      "created_at": "2024-01-15T10:30:00Z",
      "expires_at": null,
      "last_used": "2024-01-15T11:00:00Z",
      "permissions": ["create_job", "view_job"]
    }
  ]
}
```

## Job Management Endpoints

### POST /upload
Upload video file for processing.

**Auth Required:** Yes (upload_files permission)

**Request:** Multipart form data
- `file`: Video file (required)
- `language`: Language code or 'auto' (optional)
- `whisper_model`: Whisper model size (optional)
- `beam_size`: Beam search size (optional)
- `transcribe_only`: Skip visual processing (optional)
- `kf_method`: Keyframe extraction method (optional)
- `max_fps`: Maximum FPS for keyframes (optional)
- `min_scene_diff`: Minimum scene difference (optional)
- `report_style`: PDF report style (optional)

**Response (200):**
```json
{
  "message": "File uploaded successfully",
  "job_id": "job123",
  "status": "processing"
}
```

### POST /process_url
Process video from URL.

**Auth Required:** Yes (create_job permission)

**Request Body:**
```json
{
  "url": "https://youtube.com/watch?v=VIDEO_ID",
  "language": "en",
  "whisper_model": "medium",
  "beam_size": 5,
  "transcribe_only": false,
  "streaming": false,
  "kf_method": "scene",
  "max_fps": 1.0,
  "min_scene_diff": 0.45,
  "report_style": "book"
}
```

**Response (200):**
```json
{
  "message": "Processing started",
  "job_id": "job123",
  "status": "processing"
}
```

### GET /jobs
Get list of all processing jobs.

**Auth Required:** Yes (view_job permission)

**Response (200):**
```json
{
  "jobs": [
    {
      "job_id": "job123",
      "job_type": "url",
      "identifier": "https://youtube.com/watch?v=VIDEO_ID",
      "status": "completed",
      "progress": 100,
      "current_step": "completed",
      "start_time": "2024-01-15T10:30:00Z",
      "end_time": "2024-01-15T10:35:00Z",
      "duration": 300,
      "created_at": "2024-01-15T10:30:00Z",
      "last_activity": "2024-01-15T10:35:00Z"
    }
  ]
}
```

### GET /job/{job_id}
Get detailed status of a specific job.

**Auth Required:** Yes (view_job permission)

**Path Parameters:**
- `job_id`: Job ID (string)

**Response (200):**
```json
{
  "job_id": "job123",
  "status": "processing",
  "progress": 45,
  "current_step": "transcribing",
  "steps": [
    {
      "name": "download",
      "status": "completed",
      "duration": 30
    },
    {
      "name": "transcribe",
      "status": "processing",
      "duration": 120
    }
  ]
}
```

### POST /job/{job_id}/cancel
Cancel a running job.

**Auth Required:** Yes (cancel_job permission)

**Path Parameters:**
- `job_id`: Job ID (string)

**Response (200):**
```json
{
  "message": "Job cancelled successfully"
}
```

### GET /download/{job_id}/{file_type}
Download generated files from completed job.

**Auth Required:** Yes (download_files permission)

**Path Parameters:**
- `job_id`: Job ID (string)
- `file_type`: Type of file (`pdf`, `transcript`, `audio`)

**Response (200):**
Returns the requested file as binary data.

**Response (404):**
```json
{
  "error": "File not found"
}
```

## Health Monitoring Endpoints

### GET /health
Get comprehensive system health status.

**Auth Required:** No

**Response (200):**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "services": {
    "database": {
      "status": "healthy",
      "response_time_ms": 15.2
    },
    "redis": {
      "status": "healthy",
      "version": "7.0.0"
    },
    "ffmpeg": {
      "status": "healthy",
      "version": "4.4.0"
    }
  },
  "system_resources": {
    "cpu_percent": 25.5,
    "memory_percent": 45.2
  },
  "active_jobs": 2,
  "queue_size": 0
}
```

**Response (503):**
```json
{
  "status": "unhealthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "database": {
      "status": "unhealthy",
      "error_message": "Connection failed"
    }
  }
}
```

### GET /health/summary
Get simplified health overview.

**Auth Required:** No

**Response (200):**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "service_count": 10,
  "healthy_services": 9,
  "degraded_services": 1,
  "unhealthy_services": 0,
  "active_jobs": 2,
  "queue_size": 0
}
```

### GET /health/service/{service_name}
Get health status for specific service.

**Auth Required:** No

**Path Parameters:**
- `service_name`: Service name (`database`, `redis`, `ffmpeg`, etc.)

**Response (200):**
```json
{
  "name": "database",
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "details": {
    "response_time_ms": 15.2,
    "connection": "active",
    "type": "postgresql"
  }
}
```

### GET /metrics
Get Prometheus-formatted metrics.

**Auth Required:** Yes (view_metrics permission)

**Response (200):**
```
# HELP video_processing_jobs_total Total processing jobs
# TYPE video_processing_jobs_total counter
video_processing_jobs_total{job_type="url",status="completed"} 10
video_processing_jobs_total{job_type="file",status="completed"} 5
video_processing_jobs_total{job_type="url",status="failed"} 2
```

## Error Handling

All endpoints return appropriate HTTP status codes and error messages:

### Common Error Responses

**400 Bad Request:**
```json
{
  "error": "Invalid request parameters"
}
```

**401 Unauthorized:**
```json
{
  "error": "Authentication required"
}
```

**403 Forbidden:**
```json
{
  "error": "Permission denied: create_job required"
}
```

**404 Not Found:**
```json
{
  "error": "Job not found"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Internal server error"
}
```

## Rate Limiting

API endpoints may be rate limited to prevent abuse:
- **Authentication endpoints**: 5 requests per minute per IP
- **Job creation**: 10 requests per minute per user
- **File uploads**: 3 requests per minute per user

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1642248000
```

## WebSocket Events

The system supports real-time updates via WebSocket:

### Connection
Connect to `/socket.io/` with Socket.IO client.

### Events

**job_status:**
```json
{
  "job_id": "job123",
  "status": "processing",
  "progress": 45
}
```

**job_step:**
```json
{
  "job_id": "job123",
  "step": "transcribing",
  "progress": 60
}
```

**error:**
```json
{
  "job_id": "job123",
  "error": "Transcription failed"
}
```

## SDK Examples

### Python Client
```python
import requests

class VideoProcessorClient:
    def __init__(self, base_url, api_key=None, session_token=None):
        self.base_url = base_url
        self.headers = {'Content-Type': 'application/json'}
        
        if api_key:
            self.headers['X-API-Key'] = api_key
        elif session_token:
            self.headers['Authorization'] = f'Bearer {session_token}'
    
    def login(self, username, password):
        response = requests.post(
            f'{self.base_url}/auth/login',
            json={'username': username, 'password': password}
        )
        if response.ok:
            data = response.json()
            self.headers['Authorization'] = f'Bearer {data["user"]["id"]}'
            return data
        return None
    
    def create_job(self, url, **kwargs):
        response = requests.post(
            f'{self.base_url}/process_url',
            headers=self.headers,
            json={'url': url, **kwargs}
        )
        return response.json()
    
    def get_jobs(self):
        response = requests.get(
            f'{self.base_url}/jobs',
            headers=self.headers
        )
        return response.json()
    
    def get_job_status(self, job_id):
        response = requests.get(
            f'{self.base_url}/job/{job_id}',
            headers=self.headers
        )
        return response.json()

# Usage
client = VideoProcessorClient('http://localhost:5000')
client.login('admin', 'admin123')
jobs = client.get_jobs()
```

### JavaScript Client
```javascript
class VideoProcessorClient {
    constructor(baseUrl, apiKey = null, sessionToken = null) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json'
        };
        
        if (apiKey) {
            this.headers['X-API-Key'] = apiKey;
        } else if (sessionToken) {
            this.headers['Authorization'] = `Bearer ${sessionToken}`;
        }
    }
    
    async login(username, password) {
        const response = await fetch(`${this.baseUrl}/auth/login`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ username, password })
        });
        
        if (response.ok) {
            const data = await response.json();
            this.headers['Authorization'] = `Bearer ${data.user.id}`;
            return data;
        }
        return null;
    }
    
    async createJob(url, options = {}) {
        const response = await fetch(`${this.baseUrl}/process_url`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ url, ...options })
        });
        
        return await response.json();
    }
    
    async getJobs() {
        const response = await fetch(`${this.baseUrl}/jobs`, {
            headers: this.headers
        });
        
        return await response.json();
    }
    
    async getJobStatus(jobId) {
        const response = await fetch(`${this.baseUrl}/job/${jobId}`, {
            headers: this.headers
        });
        
        return await response.json();
    }
}

// Usage
const client = new VideoProcessorClient('http://localhost:5000');
await client.login('admin', 'admin123');
const jobs = await client.getJobs();
```

### cURL Examples

**Login:**
```bash
curl -X POST http://localhost:5000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

**Create Job:**
```bash
curl -X POST http://localhost:5000/process_url \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <session_token>" \
  -d '{"url": "https://youtube.com/watch?v=VIDEO_ID"}'
```

**Get Jobs:**
```bash
curl -X GET http://localhost:5000/jobs \
  -H "Authorization: Bearer <session_token>"
```

**Download PDF:**
```bash
curl -X GET http://localhost:5000/download/job123/pdf \
  -H "Authorization: Bearer <session_token>" \
  -o report.pdf
```

## Changelog

### Version 1.0.0
- Initial API release
- Authentication system
- Job management
- Health monitoring
- Interactive documentation

## Support

For API support and questions:
- **Email**: support@example.com
- **Documentation**: `/api/docs`
- **OpenAPI Spec**: `/api/openapi.json`
