# Authentication & Authorization System

This document describes the comprehensive authentication and authorization system implemented for the Video Processing application.

## Overview

The authentication system provides:
- **User Management**: Registration, login, logout, and user profiles
- **Role-Based Access Control**: Different permission levels for different user types
- **Session Management**: Secure session handling with automatic expiration
- **API Key Management**: Programmatic access with granular permissions
- **Security Features**: Password hashing, account lockout, audit logging

## User Roles & Permissions

### User Roles

#### 1. **Admin** (`admin`)
- Full system access
- Can manage users and API keys
- Access to all system health and metrics
- Can perform all operations

#### 2. **User** (`user`)
- Standard user access
- Can create, view, and edit jobs
- Can upload and download files
- Cannot access system administration features

#### 3. **Viewer** (`viewer`)
- Read-only access
- Can view jobs and download files
- Cannot create or modify content

#### 4. **API User** (`api_user`)
- Programmatic access only
- Can create jobs and access files via API
- Limited to API operations

### Permissions

| Permission | Admin | User | Viewer | API User |
|------------|-------|------|--------|----------|
| `create_job` | ✅ | ✅ | ❌ | ✅ |
| `view_job` | ✅ | ✅ | ✅ | ✅ |
| `edit_job` | ✅ | ✅ | ❌ | ❌ |
| `delete_job` | ✅ | ❌ | ❌ | ❌ |
| `cancel_job` | ✅ | ✅ | ❌ | ❌ |
| `upload_files` | ✅ | ✅ | ❌ | ✅ |
| `download_files` | ✅ | ✅ | ✅ | ✅ |
| `delete_files` | ✅ | ❌ | ❌ | ❌ |
| `view_system_health` | ✅ | ❌ | ❌ | ❌ |
| `view_metrics` | ✅ | ❌ | ❌ | ❌ |
| `manage_users` | ✅ | ❌ | ❌ | ❌ |
| `system_admin` | ✅ | ❌ | ❌ | ❌ |

## Authentication Endpoints

### Web Authentication

#### 1. **Login**
- **URL**: `/auth/login`
- **Method**: POST
- **Description**: Authenticate user and create session
- **Body**: `{"username": "string", "password": "string"}`
- **Response**: User information and session token

#### 2. **Logout**
- **URL**: `/auth/logout`
- **Method**: POST
- **Description**: Logout user and invalidate session
- **Auth Required**: Yes
- **Response**: Success message

#### 3. **Get Current User**
- **URL**: `/auth/me`
- **Method**: GET
- **Description**: Get current user information
- **Auth Required**: Yes
- **Response**: User details and permissions

#### 4. **Register**
- **URL**: `/auth/register`
- **Method**: POST
- **Description**: Register new user
- **Body**: `{"username": "string", "email": "string", "password": "string"}`
- **Response**: Created user information

### API Key Management

#### 1. **Create API Key**
- **URL**: `/auth/api-keys`
- **Method**: POST
- **Description**: Create new API key
- **Auth Required**: Yes (manage_users permission)
- **Body**: `{"key_name": "string", "permissions": ["string"], "expires_days": number}`

#### 2. **List API Keys**
- **URL**: `/auth/api-keys`
- **Method**: GET
- **Description**: List user's API keys
- **Auth Required**: Yes (manage_users permission)
- **Response**: Array of API key information

## Security Features

### Password Security
- **Bcrypt Hashing**: Passwords are hashed using bcrypt with salt
- **Minimum Length**: 6 characters required
- **No Plain Text Storage**: Passwords are never stored in plain text

### Account Security
- **Failed Login Protection**: Account locks after 5 failed attempts
- **Lockout Duration**: 15 minutes lockout period
- **Session Timeout**: Sessions expire after 1 hour of inactivity
- **Secure Sessions**: Session tokens are cryptographically secure

### API Security
- **API Key Authentication**: Secure API keys for programmatic access
- **Permission-Based Access**: Granular permissions for API operations
- **Key Expiration**: Optional expiration dates for API keys
- **Usage Tracking**: Last used timestamps for API keys

### Audit Logging
- **Login Attempts**: All login attempts are logged
- **Permission Checks**: Permission denials are logged
- **API Usage**: API key usage is tracked
- **User Actions**: User operations are audited

## Usage Examples

### Web Authentication

#### Login
```javascript
const response = await fetch('/auth/login', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        username: 'admin',
        password: 'admin123'
    })
});

const data = await response.json();
if (response.ok) {
    console.log('Login successful:', data.user);
} else {
    console.error('Login failed:', data.error);
}
```

#### Logout
```javascript
const response = await fetch('/auth/logout', {
    method: 'POST'
});

if (response.ok) {
    console.log('Logout successful');
    // Clear local storage and redirect
    localStorage.removeItem('user');
    window.location.href = '/';
}
```

### API Authentication

#### Using Session Token
```bash
curl -H "Authorization: Bearer <session_token>" \
     -H "Content-Type: application/json" \
     http://localhost:5000/jobs
```

#### Using API Key
```bash
curl -H "X-API-Key: <api_key>" \
     -H "Content-Type: application/json" \
     http://localhost:5000/jobs
```

#### Create Job with Authentication
```bash
curl -H "X-API-Key: <api_key>" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/video.mp4"}' \
     http://localhost:5000/process_url
```

### Python API Client
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
    
    def create_job(self, url):
        response = requests.post(
            f'{self.base_url}/process_url',
            headers=self.headers,
            json={'url': url}
        )
        return response.json()
    
    def get_jobs(self):
        response = requests.get(
            f'{self.base_url}/jobs',
            headers=self.headers
        )
        return response.json()

# Usage
client = VideoProcessorClient('http://localhost:5000', api_key='your_api_key')
jobs = client.get_jobs()
```

## Default Credentials

When the system starts for the first time, a default admin user is created:

- **Username**: `admin`
- **Password**: `admin123`
- **Email**: `admin@example.com`
- **Role**: `admin`

**⚠️ Important**: Change the default password in production!

## Environment Variables

Configure authentication using environment variables:

```bash
# JWT Secret (auto-generated if not provided)
JWT_SECRET=your-secret-key-here

# Session Configuration
SESSION_TIMEOUT=3600  # 1 hour in seconds
MAX_FAILED_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION=900  # 15 minutes in seconds

# Default Admin Credentials
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=admin123

# Flask Secret Key
SECRET_KEY=your-flask-secret-key
```

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    preferences TEXT
);
```

### User Sessions Table
```sql
CREATE TABLE user_sessions (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) REFERENCES users(id),
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE
);
```

### API Keys Table
```sql
CREATE TABLE api_keys (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) REFERENCES users(id),
    key_name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    permissions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    last_used TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

## Integration with Existing Endpoints

All existing endpoints now support authentication:

### Protected Endpoints
- `/upload` - Requires `upload_files` permission
- `/process_url` - Requires `create_job` permission
- `/jobs` - Requires `view_job` permission
- `/metrics` - Requires `view_metrics` permission
- `/health-dashboard` - Requires `view_system_health` permission

### Public Endpoints
- `/health` - Public health check
- `/health/summary` - Public health summary
- `/auth/login` - Login endpoint
- `/auth/register` - Registration endpoint

## Troubleshooting

### Common Issues

#### 1. **Authentication Required Error**
- **Cause**: Endpoint requires authentication but no token provided
- **Solution**: Login first or provide API key

#### 2. **Permission Denied Error**
- **Cause**: User doesn't have required permission
- **Solution**: Check user role and permissions

#### 3. **Session Expired Error**
- **Cause**: Session token has expired
- **Solution**: Login again to get new session

#### 4. **Account Locked Error**
- **Cause**: Too many failed login attempts
- **Solution**: Wait for lockout period or contact admin

### Debug Mode

Enable debug logging for authentication issues:

```bash
export LOG_LEVEL=DEBUG
export SQL_ECHO=true
python web_app.py
```

## Best Practices

### Security
1. **Use HTTPS** in production
2. **Change default passwords** immediately
3. **Regular password updates** for admin accounts
4. **Monitor failed login attempts**
5. **Use strong API keys** with limited permissions

### User Management
1. **Principle of Least Privilege** - Give users minimum required permissions
2. **Regular permission reviews** - Audit user permissions periodically
3. **Account lifecycle management** - Deactivate unused accounts
4. **API key rotation** - Regularly rotate API keys

### Monitoring
1. **Monitor authentication logs** for suspicious activity
2. **Track API key usage** patterns
3. **Alert on failed login attempts**
4. **Monitor session activity**

## Migration Guide

If you're upgrading from a version without authentication:

1. **Database Migration**: Run the database initialization to create auth tables
2. **Default Admin**: The system will create a default admin user
3. **Update Clients**: Update API clients to include authentication
4. **Test Access**: Verify all endpoints work with authentication

## API Documentation

For complete API documentation, see the OpenAPI specification (to be implemented) or use the interactive API explorer at `/api/docs` (planned feature).
