"""
API Documentation system with OpenAPI/Swagger integration.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from flask import Blueprint, render_template, jsonify, request
from .auth import Permission, UserRole


class HTTPMethod(str, Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class APIEndpoint:
    """API endpoint definition."""
    path: str
    method: HTTPMethod
    summary: str
    description: str
    tags: List[str]
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]]
    responses: Dict[str, Dict[str, Any]]
    auth_required: bool
    permissions: List[str]
    examples: List[Dict[str, Any]]


class APIDocumentation:
    """API documentation generator."""
    
    def __init__(self):
        self.endpoints: List[APIEndpoint] = []
        self._initialize_endpoints()
    
    def _initialize_endpoints(self):
        """Initialize all API endpoints."""
        
        # Authentication endpoints
        self.endpoints.extend([
            APIEndpoint(
                path="/auth/login",
                method=HTTPMethod.POST,
                summary="User Login",
                description="Authenticate user and create session",
                tags=["Authentication"],
                parameters=[],
                request_body={
                    "type": "object",
                    "required": ["username", "password"],
                    "properties": {
                        "username": {"type": "string", "description": "Username"},
                        "password": {"type": "string", "description": "Password"}
                    }
                },
                responses={
                    "200": {
                        "description": "Login successful",
                        "content": {
                            "application/json": {
                                "example": {
                                    "message": "Login successful",
                                    "user": {
                                        "id": "user123",
                                        "username": "admin",
                                        "role": "admin",
                                        "permissions": ["create_job", "view_job", "manage_users"]
                                    }
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "Invalid credentials",
                        "content": {
                            "application/json": {
                                "example": {"error": "Invalid username or password"}
                            }
                        }
                    }
                },
                auth_required=False,
                permissions=[],
                examples=[
                    {
                        "name": "Admin Login",
                        "request": {"username": "admin", "password": "admin123"},
                        "response": {
                            "message": "Login successful",
                            "user": {"username": "admin", "role": "admin"}
                        }
                    }
                ]
            ),
            
            APIEndpoint(
                path="/auth/logout",
                method=HTTPMethod.POST,
                summary="User Logout",
                description="Logout user and invalidate session",
                tags=["Authentication"],
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "Logout successful",
                        "content": {
                            "application/json": {
                                "example": {"message": "Logout successful"}
                            }
                        }
                    }
                },
                auth_required=True,
                permissions=[],
                examples=[]
            ),
            
            APIEndpoint(
                path="/auth/me",
                method=HTTPMethod.GET,
                summary="Get Current User",
                description="Get current user information and permissions",
                tags=["Authentication"],
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "User information",
                        "content": {
                            "application/json": {
                                "example": {
                                    "user": {
                                        "id": "user123",
                                        "username": "admin",
                                        "role": "admin",
                                        "permissions": ["create_job", "view_job", "manage_users"]
                                    }
                                }
                            }
                        }
                    }
                },
                auth_required=True,
                permissions=[],
                examples=[]
            ),
            
            APIEndpoint(
                path="/auth/register",
                method=HTTPMethod.POST,
                summary="Register User",
                description="Register new user account",
                tags=["Authentication"],
                parameters=[],
                request_body={
                    "type": "object",
                    "required": ["username", "email", "password"],
                    "properties": {
                        "username": {"type": "string", "description": "Username (min 3 chars)"},
                        "email": {"type": "string", "format": "email", "description": "Email address"},
                        "password": {"type": "string", "description": "Password (min 6 chars)"}
                    }
                },
                responses={
                    "201": {
                        "description": "User created successfully",
                        "content": {
                            "application/json": {
                                "example": {
                                    "message": "User created successfully",
                                    "user": {
                                        "id": "user123",
                                        "username": "newuser",
                                        "email": "user@example.com",
                                        "role": "user"
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Validation error",
                        "content": {
                            "application/json": {
                                "example": {"error": "Username must be at least 3 characters"}
                            }
                        }
                    }
                },
                auth_required=False,
                permissions=[],
                examples=[]
            ),
            
            APIEndpoint(
                path="/auth/api-keys",
                method=HTTPMethod.POST,
                summary="Create API Key",
                description="Create new API key for programmatic access",
                tags=["Authentication"],
                parameters=[],
                request_body={
                    "type": "object",
                    "required": ["key_name"],
                    "properties": {
                        "key_name": {"type": "string", "description": "Name for the API key"},
                        "permissions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of permissions for the API key"
                        },
                        "expires_days": {"type": "integer", "description": "Expiration in days (optional)"}
                    }
                },
                responses={
                    "201": {
                        "description": "API key created",
                        "content": {
                            "application/json": {
                                "example": {
                                    "message": "API key created successfully",
                                    "api_key": "sk-1234567890abcdef",
                                    "key_name": "My API Key"
                                }
                            }
                        }
                    }
                },
                auth_required=True,
                permissions=[Permission.MANAGE_USERS.value],
                examples=[]
            ),
            
            APIEndpoint(
                path="/auth/api-keys",
                method=HTTPMethod.GET,
                summary="List API Keys",
                description="Get list of user's API keys",
                tags=["Authentication"],
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "List of API keys",
                        "content": {
                            "application/json": {
                                "example": {
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
                            }
                        }
                    }
                },
                auth_required=True,
                permissions=[Permission.MANAGE_USERS.value],
                examples=[]
            )
        ])
        
        # Job management endpoints
        self.endpoints.extend([
            APIEndpoint(
                path="/upload",
                method=HTTPMethod.POST,
                summary="Upload Video File",
                description="Upload video file for processing",
                tags=["Jobs"],
                parameters=[],
                request_body={
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {"type": "string", "format": "binary", "description": "Video file"},
                        "language": {"type": "string", "description": "Language code or 'auto'"},
                        "whisper_model": {"type": "string", "description": "Whisper model size"},
                        "beam_size": {"type": "integer", "description": "Beam search size"},
                        "transcribe_only": {"type": "boolean", "description": "Skip visual processing"},
                        "kf_method": {"type": "string", "description": "Keyframe extraction method"},
                        "max_fps": {"type": "number", "description": "Maximum FPS for keyframes"},
                        "min_scene_diff": {"type": "number", "description": "Minimum scene difference"},
                        "report_style": {"type": "string", "description": "PDF report style"}
                    }
                },
                responses={
                    "200": {
                        "description": "File uploaded successfully",
                        "content": {
                            "application/json": {
                                "example": {
                                    "message": "File uploaded successfully",
                                    "job_id": "job123",
                                    "status": "processing"
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Upload failed",
                        "content": {
                            "application/json": {
                                "example": {"error": "No file provided"}
                            }
                        }
                    }
                },
                auth_required=True,
                permissions=[Permission.UPLOAD_FILES.value],
                examples=[]
            ),
            
            APIEndpoint(
                path="/process_url",
                method=HTTPMethod.POST,
                summary="Process Video URL",
                description="Process video from URL",
                tags=["Jobs"],
                parameters=[],
                request_body={
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": {"type": "string", "format": "uri", "description": "Video URL"},
                        "language": {"type": "string", "description": "Language code or 'auto'"},
                        "whisper_model": {"type": "string", "description": "Whisper model size"},
                        "beam_size": {"type": "integer", "description": "Beam search size"},
                        "transcribe_only": {"type": "boolean", "description": "Skip visual processing"},
                        "streaming": {"type": "boolean", "description": "Use streaming mode"},
                        "kf_method": {"type": "string", "description": "Keyframe extraction method"},
                        "max_fps": {"type": "number", "description": "Maximum FPS for keyframes"},
                        "min_scene_diff": {"type": "number", "description": "Minimum scene difference"},
                        "report_style": {"type": "string", "description": "PDF report style"}
                    }
                },
                responses={
                    "200": {
                        "description": "Processing started",
                        "content": {
                            "application/json": {
                                "example": {
                                    "message": "Processing started",
                                    "job_id": "job123",
                                    "status": "processing"
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid URL or parameters",
                        "content": {
                            "application/json": {
                                "example": {"error": "Invalid URL provided"}
                            }
                        }
                    }
                },
                auth_required=True,
                permissions=[Permission.CREATE_JOB.value],
                examples=[
                    {
                        "name": "Process YouTube Video",
                        "request": {
                            "url": "https://youtube.com/watch?v=VIDEO_ID",
                            "language": "en",
                            "whisper_model": "medium",
                            "transcribe_only": False
                        },
                        "response": {
                            "job_id": "job123",
                            "status": "processing"
                        }
                    }
                ]
            ),
            
            APIEndpoint(
                path="/jobs",
                method=HTTPMethod.GET,
                summary="List Jobs",
                description="Get list of all processing jobs",
                tags=["Jobs"],
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "List of jobs",
                        "content": {
                            "application/json": {
                                "example": {
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
                            }
                        }
                    }
                },
                auth_required=True,
                permissions=[Permission.VIEW_JOB.value],
                examples=[]
            ),
            
            APIEndpoint(
                path="/job/{job_id}",
                method=HTTPMethod.GET,
                summary="Get Job Status",
                description="Get detailed status of a specific job",
                tags=["Jobs"],
                parameters=[
                    {
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Job ID"
                    }
                ],
                request_body=None,
                responses={
                    "200": {
                        "description": "Job status",
                        "content": {
                            "application/json": {
                                "example": {
                                    "job_id": "job123",
                                    "status": "processing",
                                    "progress": 45,
                                    "current_step": "transcribing",
                                    "steps": [
                                        {"name": "download", "status": "completed", "duration": 30},
                                        {"name": "transcribe", "status": "processing", "duration": 120}
                                    ]
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Job not found",
                        "content": {
                            "application/json": {
                                "example": {"error": "Job not found"}
                            }
                        }
                    }
                },
                auth_required=True,
                permissions=[Permission.VIEW_JOB.value],
                examples=[]
            ),
            
            APIEndpoint(
                path="/job/{job_id}/cancel",
                method=HTTPMethod.POST,
                summary="Cancel Job",
                description="Cancel a running job",
                tags=["Jobs"],
                parameters=[
                    {
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Job ID"
                    }
                ],
                request_body=None,
                responses={
                    "200": {
                        "description": "Job cancelled",
                        "content": {
                            "application/json": {
                                "example": {"message": "Job cancelled successfully"}
                            }
                        }
                    },
                    "400": {
                        "description": "Job cannot be cancelled",
                        "content": {
                            "application/json": {
                                "example": {"error": "Job cannot be cancelled"}
                            }
                        }
                    }
                },
                auth_required=True,
                permissions=[Permission.CANCEL_JOB.value],
                examples=[]
            ),
            
            APIEndpoint(
                path="/download/{job_id}/{file_type}",
                method=HTTPMethod.GET,
                summary="Download File",
                description="Download generated files from completed job",
                tags=["Jobs"],
                parameters=[
                    {
                        "name": "job_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Job ID"
                    },
                    {
                        "name": "file_type",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string", "enum": ["pdf", "transcript", "audio"]},
                        "description": "Type of file to download"
                    }
                ],
                request_body=None,
                responses={
                    "200": {
                        "description": "File download",
                        "content": {
                            "application/pdf": {"schema": {"type": "string", "format": "binary"}},
                            "text/plain": {"schema": {"type": "string"}},
                            "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                        }
                    },
                    "404": {
                        "description": "File not found",
                        "content": {
                            "application/json": {
                                "example": {"error": "File not found"}
                            }
                        }
                    }
                },
                auth_required=True,
                permissions=[Permission.DOWNLOAD_FILES.value],
                examples=[]
            )
        ])
        
        # Health and monitoring endpoints
        self.endpoints.extend([
            APIEndpoint(
                path="/health",
                method=HTTPMethod.GET,
                summary="System Health Check",
                description="Get comprehensive system health status",
                tags=["Health"],
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "System healthy",
                        "content": {
                            "application/json": {
                                "example": {
                                    "status": "healthy",
                                    "timestamp": "2024-01-15T10:30:00Z",
                                    "uptime_seconds": 3600,
                                    "version": "1.0.0",
                                    "services": {
                                        "database": {"status": "healthy", "response_time_ms": 15.2},
                                        "redis": {"status": "healthy", "version": "7.0.0"},
                                        "ffmpeg": {"status": "healthy", "version": "4.4.0"}
                                    },
                                    "system_resources": {
                                        "cpu_percent": 25.5,
                                        "memory_percent": 45.2
                                    },
                                    "active_jobs": 2,
                                    "queue_size": 0
                                }
                            }
                        }
                    },
                    "503": {
                        "description": "System unhealthy",
                        "content": {
                            "application/json": {
                                "example": {
                                    "status": "unhealthy",
                                    "timestamp": "2024-01-15T10:30:00Z",
                                    "services": {
                                        "database": {"status": "unhealthy", "error_message": "Connection failed"}
                                    }
                                }
                            }
                        }
                    }
                },
                auth_required=False,
                permissions=[],
                examples=[]
            ),
            
            APIEndpoint(
                path="/health/summary",
                method=HTTPMethod.GET,
                summary="Health Summary",
                description="Get simplified health overview",
                tags=["Health"],
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "Health summary",
                        "content": {
                            "application/json": {
                                "example": {
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
                            }
                        }
                    }
                },
                auth_required=False,
                permissions=[],
                examples=[]
            ),
            
            APIEndpoint(
                path="/health/service/{service_name}",
                method=HTTPMethod.GET,
                summary="Service Health",
                description="Get health status for specific service",
                tags=["Health"],
                parameters=[
                    {
                        "name": "service_name",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Service name (database, redis, ffmpeg, etc.)"
                    }
                ],
                request_body=None,
                responses={
                    "200": {
                        "description": "Service health",
                        "content": {
                            "application/json": {
                                "example": {
                                    "name": "database",
                                    "status": "healthy",
                                    "timestamp": "2024-01-15T10:30:00Z",
                                    "details": {
                                        "response_time_ms": 15.2,
                                        "connection": "active",
                                        "type": "postgresql"
                                    }
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Service not found",
                        "content": {
                            "application/json": {
                                "example": {"error": "Service not found"}
                            }
                        }
                    }
                },
                auth_required=False,
                permissions=[],
                examples=[]
            ),
            
            APIEndpoint(
                path="/metrics",
                method=HTTPMethod.GET,
                summary="Prometheus Metrics",
                description="Get Prometheus-formatted metrics",
                tags=["Monitoring"],
                parameters=[],
                request_body=None,
                responses={
                    "200": {
                        "description": "Prometheus metrics",
                        "content": {
                            "text/plain": {
                                "example": "# HELP video_processing_jobs_total Total processing jobs\n# TYPE video_processing_jobs_total counter\nvideo_processing_jobs_total{job_type=\"url\",status=\"completed\"} 10"
                            }
                        }
                    }
                },
                auth_required=True,
                permissions=[Permission.VIEW_METRICS.value],
                examples=[]
            )
        ])
    
    def get_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification."""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Video Processing API",
                "description": "Comprehensive API for video transcription and PDF generation",
                "version": "1.0.0",
                "contact": {
                    "name": "API Support",
                    "email": "support@example.com"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:5000",
                    "description": "Development server"
                },
                {
                    "url": "https://api.example.com",
                    "description": "Production server"
                }
            ],
            "tags": [
                {"name": "Authentication", "description": "User authentication and authorization"},
                {"name": "Jobs", "description": "Video processing job management"},
                {"name": "Health", "description": "System health monitoring"},
                {"name": "Monitoring", "description": "System metrics and monitoring"}
            ],
            "paths": self._generate_paths(),
            "components": {
                "securitySchemes": {
                    "sessionAuth": {
                        "type": "apiKey",
                        "in": "cookie",
                        "name": "session_token",
                        "description": "Session token from login"
                    },
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT",
                        "description": "Bearer token authentication"
                    },
                    "apiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key",
                        "description": "API key for programmatic access"
                    }
                },
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "username": {"type": "string"},
                            "role": {"type": "string", "enum": ["admin", "user", "viewer", "api_user"]},
                            "permissions": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "Job": {
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string"},
                            "job_type": {"type": "string"},
                            "identifier": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "processing", "completed", "failed", "cancelled"]},
                            "progress": {"type": "integer", "minimum": 0, "maximum": 100},
                            "current_step": {"type": "string"},
                            "start_time": {"type": "string", "format": "date-time"},
                            "end_time": {"type": "string", "format": "date-time"},
                            "duration": {"type": "integer"},
                            "created_at": {"type": "string", "format": "date-time"},
                            "last_activity": {"type": "string", "format": "date-time"}
                        }
                    },
                    "Error": {
                        "type": "object",
                        "properties": {
                            "error": {"type": "string"},
                            "message": {"type": "string"},
                            "code": {"type": "string"}
                        }
                    }
                }
            }
        }
    
    def _generate_paths(self) -> Dict[str, Any]:
        """Generate OpenAPI paths from endpoints."""
        paths = {}
        
        for endpoint in self.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}
            
            operation = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "responses": endpoint.responses
            }
            
            if endpoint.parameters:
                operation["parameters"] = endpoint.parameters
            
            if endpoint.request_body:
                operation["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": endpoint.request_body
                        }
                    }
                }
            
            if endpoint.auth_required:
                operation["security"] = [
                    {"sessionAuth": []},
                    {"bearerAuth": []},
                    {"apiKeyAuth": []}
                ]
            
            if endpoint.examples:
                operation["examples"] = {
                    example["name"]: {
                        "summary": example["name"],
                        "value": example
                    }
                    for example in endpoint.examples
                }
            
            paths[endpoint.path][endpoint.method.value.lower()] = operation
        
        return paths
    
    def get_endpoints_by_tag(self, tag: str) -> List[APIEndpoint]:
        """Get endpoints filtered by tag."""
        return [ep for ep in self.endpoints if tag in ep.tags]
    
    def get_endpoints_by_permission(self, permission: str) -> List[APIEndpoint]:
        """Get endpoints filtered by permission."""
        return [ep for ep in self.endpoints if permission in ep.permissions]


# Global API documentation instance
api_docs = APIDocumentation()


def create_api_docs_blueprint() -> Blueprint:
    """Create Flask blueprint for API documentation."""
    bp = Blueprint('api_docs', __name__, url_prefix='/api')
    
    @bp.route('/docs')
    def api_documentation():
        """Interactive API documentation page."""
        return render_template('api_docs.html', endpoints=api_docs.endpoints)
    
    @bp.route('/openapi.json')
    def openapi_spec():
        """OpenAPI specification endpoint."""
        return jsonify(api_docs.get_openapi_spec())
    
    @bp.route('/endpoints')
    def list_endpoints():
        """List all API endpoints."""
        endpoints_data = []
        for endpoint in api_docs.endpoints:
            endpoints_data.append({
                'path': endpoint.path,
                'method': endpoint.method.value,
                'summary': endpoint.summary,
                'tags': endpoint.tags,
                'auth_required': endpoint.auth_required,
                'permissions': endpoint.permissions
            })
        
        return jsonify({
            'endpoints': endpoints_data,
            'total': len(endpoints_data)
        })
    
    @bp.route('/endpoints/<tag>')
    def endpoints_by_tag(tag: str):
        """Get endpoints by tag."""
        filtered_endpoints = api_docs.get_endpoints_by_tag(tag)
        return jsonify({
            'tag': tag,
            'endpoints': [asdict(ep) for ep in filtered_endpoints],
            'count': len(filtered_endpoints)
        })
    
    return bp
