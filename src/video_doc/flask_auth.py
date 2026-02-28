"""
Flask authentication decorators and middleware for video processing application.
"""

import functools
from typing import Optional, Callable, Any
from flask import request, jsonify, session, g, current_app
from werkzeug.exceptions import Unauthorized, Forbidden

from .auth import auth_manager, UserSession, Permission, get_current_user
from .monitoring import get_logger, audit_logger
from .error_handling import ProcessingError, ErrorSeverity

logger = get_logger(__name__)


def get_client_ip() -> str:
    """Get client IP address from request."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    elif request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    else:
        return request.remote_addr


def get_user_agent() -> str:
    """Get user agent from request."""
    return request.headers.get('User-Agent', '')


def extract_auth_token() -> Optional[str]:
    """Extract authentication token from request."""
    # Check Authorization header
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header[7:]
    
    # Check session cookie
    if 'session_token' in session:
        return session['session_token']
    
    # Check API key header
    api_key = request.headers.get('X-API-Key')
    if api_key:
        return api_key
    
    return None


def authenticate_request() -> Optional[UserSession]:
    """Authenticate the current request."""
    token = extract_auth_token()
    if not token:
        return None
    
    # Try session token first
    user_session = auth_manager.validate_session(token)
    if user_session:
        return user_session
    
    # Try API key
    api_user_info = auth_manager.validate_api_key(token)
    if api_user_info:
        # Convert API key info to UserSession-like object
        return UserSession(
            user_id=api_user_info['user_id'],
            username=api_user_info['username'],
            role=api_user_info['role'],
            permissions=api_user_info['permissions'],
            session_id=api_user_info['api_key_id'],
            created_at=None,
            expires_at=None
        )
    
    return None


def require_auth(permission: Optional[Permission] = None):
    """Decorator to require authentication and optionally specific permission."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Authenticate user
            user_session = authenticate_request()
            if not user_session:
                audit_logger.log_operation(
                    operation="auth_required_failed",
                    resource_type="endpoint",
                    resource_id=request.endpoint,
                    ip_address=get_client_ip(),
                    user_agent=get_user_agent(),
                    metadata={'reason': 'no_token'}
                )
                return jsonify({'error': 'Authentication required'}), 401
            
            # Check permission if specified
            if permission and not auth_manager.has_permission(user_session, permission):
                audit_logger.log_operation(
                    operation="permission_denied",
                    resource_type="endpoint",
                    resource_id=request.endpoint,
                    user_id=user_session.user_id,
                    ip_address=get_client_ip(),
                    user_agent=get_user_agent(),
                    metadata={'required_permission': permission.value}
                )
                return jsonify({'error': f'Permission denied: {permission.value} required'}), 403
            
            # Store user session in Flask g for access in route handlers
            g.current_user = user_session
            
            # Log successful authentication
            audit_logger.log_operation(
                operation="authenticated_access",
                resource_type="endpoint",
                resource_id=request.endpoint,
                user_id=user_session.user_id,
                ip_address=get_client_ip(),
                user_agent=get_user_agent()
            )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(role: str):
    """Decorator to require specific user role."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_session = authenticate_request()
            if not user_session:
                return jsonify({'error': 'Authentication required'}), 401
            
            if user_session.role.value != role:
                audit_logger.log_operation(
                    operation="role_denied",
                    resource_type="endpoint",
                    resource_id=request.endpoint,
                    user_id=user_session.user_id,
                    ip_address=get_client_ip(),
                    user_agent=get_user_agent(),
                    metadata={'required_role': role, 'user_role': user_session.role.value}
                )
                return jsonify({'error': f'Role {role} required'}), 403
            
            g.current_user = user_session
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def optional_auth(func: Callable) -> Callable:
    """Decorator for optional authentication - sets g.current_user if authenticated."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        user_session = authenticate_request()
        if user_session:
            g.current_user = user_session
        return func(*args, **kwargs)
    
    return wrapper


def get_current_user_session() -> Optional[UserSession]:
    """Get current user session from Flask g."""
    return getattr(g, 'current_user', None)


def login_user(username: str, password: str) -> Optional[UserSession]:
    """Login user and create session."""
    user_session = auth_manager.authenticate_user(
        username=username,
        password=password,
        ip_address=get_client_ip()
    )
    
    if user_session:
        # Store session token in Flask session
        session['session_token'] = user_session.session_id
        session['user_id'] = user_session.user_id
        session['username'] = user_session.username
        session['role'] = user_session.role.value
        
        logger.info(f"User logged in: {username}")
        return user_session
    
    return None


def logout_user() -> bool:
    """Logout current user."""
    session_token = session.get('session_token')
    if session_token:
        success = auth_manager.logout_user(session_token)
        if success:
            # Clear Flask session
            session.clear()
            logger.info("User logged out")
            return True
    
    return False


def create_user_endpoints(app):
    """Create user management endpoints."""
    
    @app.route('/auth/login', methods=['POST'])
    def login():
        """Login endpoint."""
        try:
            data = request.get_json()
            if not data or 'username' not in data or 'password' not in data:
                return jsonify({'error': 'Username and password required'}), 400
            
            user_session = login_user(data['username'], data['password'])
            if user_session:
                return jsonify({
                    'message': 'Login successful',
                    'user': {
                        'id': user_session.user_id,
                        'username': user_session.username,
                        'role': user_session.role.value,
                        'permissions': [p.value for p in user_session.permissions]
                    }
                })
            else:
                return jsonify({'error': 'Invalid username or password'}), 401
                
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return jsonify({'error': 'Login failed'}), 500
    
    @app.route('/auth/logout', methods=['POST'])
    @require_auth()
    def logout():
        """Logout endpoint."""
        try:
            success = logout_user()
            if success:
                return jsonify({'message': 'Logout successful'})
            else:
                return jsonify({'error': 'Logout failed'}), 500
                
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return jsonify({'error': 'Logout failed'}), 500
    
    @app.route('/auth/me', methods=['GET'])
    @require_auth()
    def get_current_user_info():
        """Get current user information."""
        try:
            user_session = get_current_user_session()
            if user_session:
                return jsonify({
                    'user': {
                        'id': user_session.user_id,
                        'username': user_session.username,
                        'role': user_session.role.value,
                        'permissions': [p.value for p in user_session.permissions]
                    }
                })
            else:
                return jsonify({'error': 'User not found'}), 404
                
        except Exception as e:
            logger.error(f"Get user info error: {str(e)}")
            return jsonify({'error': 'Failed to get user info'}), 500
    
    @app.route('/auth/register', methods=['POST'])
    def register():
        """Register new user endpoint."""
        try:
            data = request.get_json()
            if not data or 'username' not in data or 'email' not in data or 'password' not in data:
                return jsonify({'error': 'Username, email, and password required'}), 400
            
            # Validate input
            if len(data['username']) < 3:
                return jsonify({'error': 'Username must be at least 3 characters'}), 400
            
            if len(data['password']) < 6:
                return jsonify({'error': 'Password must be at least 6 characters'}), 400
            
            # Create user
            user = auth_manager.create_user(
                username=data['username'],
                email=data['email'],
                password=data['password'],
                role=data.get('role', 'user')
            )
            
            return jsonify({
                'message': 'User created successfully',
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role
                }
            }), 201
            
        except ProcessingError as e:
            return jsonify({'error': e.message}), 400
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            return jsonify({'error': 'Registration failed'}), 500
    
    @app.route('/auth/api-keys', methods=['POST'])
    @require_auth(Permission.MANAGE_USERS)
    def create_api_key():
        """Create API key endpoint."""
        try:
            data = request.get_json()
            if not data or 'key_name' not in data:
                return jsonify({'error': 'Key name required'}), 400
            
            user_session = get_current_user_session()
            permissions = [Permission(p) for p in data.get('permissions', [])]
            expires_days = data.get('expires_days')
            
            api_key = auth_manager.create_api_key(
                user_id=user_session.user_id,
                key_name=data['key_name'],
                permissions=permissions,
                expires_days=expires_days
            )
            
            return jsonify({
                'message': 'API key created successfully',
                'api_key': api_key,
                'key_name': data['key_name']
            }), 201
            
        except ProcessingError as e:
            return jsonify({'error': e.message}), 400
        except Exception as e:
            logger.error(f"API key creation error: {str(e)}")
            return jsonify({'error': 'API key creation failed'}), 500
    
    @app.route('/auth/api-keys', methods=['GET'])
    @require_auth(Permission.MANAGE_USERS)
    def list_api_keys():
        """List user's API keys (metadata only; the actual key is never returned)."""
        try:
            user_session = get_current_user_session()
            api_keys = auth_manager.list_api_keys(user_session.user_id)
            return jsonify({'api_keys': api_keys})
            
        except Exception as e:
            logger.error(f"List API keys error: {str(e)}")
            return jsonify({'error': 'Failed to list API keys'}), 500


def init_auth_system(app):
    """Initialize authentication system for Flask app."""
    # Set secret key for sessions
    if not app.config.get('SECRET_KEY'):
        app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'video-doc-secret-key-2024')
    
    # Create user management endpoints
    create_user_endpoints(app)
    
    # Create default admin user
    try:
        from .auth import create_default_admin
        create_default_admin()
    except Exception as e:
        logger.error(f"Failed to create default admin: {str(e)}")
    
    logger.info("Authentication system initialized")
