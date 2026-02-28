"""
Comprehensive authentication and authorization system for video processing application.
"""

import os
import hashlib
import secrets
import jwt
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import bcrypt

from .database import get_db_session, Base
from .monitoring import get_logger, audit_logger
from .error_handling import ProcessingError, ErrorSeverity
from sqlalchemy import Column, String, DateTime, Boolean, Text, Integer, ForeignKey
from sqlalchemy.orm import relationship

logger = get_logger(__name__)


class UserRole(str, Enum):
    """User roles with different permission levels."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_USER = "api_user"


class Permission(str, Enum):
    """System permissions."""
    # Job permissions
    CREATE_JOB = "create_job"
    VIEW_JOB = "view_job"
    EDIT_JOB = "edit_job"
    DELETE_JOB = "delete_job"
    CANCEL_JOB = "cancel_job"
    
    # System permissions
    VIEW_SYSTEM_HEALTH = "view_system_health"
    VIEW_METRICS = "view_metrics"
    MANAGE_USERS = "manage_users"
    SYSTEM_ADMIN = "system_admin"
    
    # File permissions
    UPLOAD_FILES = "upload_files"
    DOWNLOAD_FILES = "download_files"
    DELETE_FILES = "delete_files"


# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.CREATE_JOB,
        Permission.VIEW_JOB,
        Permission.EDIT_JOB,
        Permission.DELETE_JOB,
        Permission.CANCEL_JOB,
        Permission.VIEW_SYSTEM_HEALTH,
        Permission.VIEW_METRICS,
        Permission.MANAGE_USERS,
        Permission.SYSTEM_ADMIN,
        Permission.UPLOAD_FILES,
        Permission.DOWNLOAD_FILES,
        Permission.DELETE_FILES,
    ],
    UserRole.USER: [
        Permission.CREATE_JOB,
        Permission.VIEW_JOB,
        Permission.EDIT_JOB,
        Permission.CANCEL_JOB,
        Permission.UPLOAD_FILES,
        Permission.DOWNLOAD_FILES,
    ],
    UserRole.VIEWER: [
        Permission.VIEW_JOB,
        Permission.DOWNLOAD_FILES,
    ],
    UserRole.API_USER: [
        Permission.CREATE_JOB,
        Permission.VIEW_JOB,
        Permission.UPLOAD_FILES,
        Permission.DOWNLOAD_FILES,
    ],
}


@dataclass
class UserSession:
    """User session data."""
    user_id: str
    username: str
    role: UserRole
    permissions: List[Permission]
    session_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class User(Base):
    """User model for authentication."""
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default=UserRole.USER.value)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    
    # User preferences
    preferences = Column(Text)  # JSON string
    
    # Relationships
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")


class UserSession(Base):
    """User session model."""
    __tablename__ = 'user_sessions'
    
    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    ip_address = Column(String(45))  # IPv6 compatible
    user_agent = Column(Text)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class APIKey(Base):
    """API key model for programmatic access."""
    __tablename__ = 'api_keys'
    
    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    key_name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False)
    permissions = Column(Text)  # JSON string of permissions
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")


class AuthenticationManager:
    """Main authentication manager."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.jwt_secret = os.getenv('JWT_SECRET', secrets.token_urlsafe(32))
        self.jwt_algorithm = 'HS256'
        self.session_timeout = int(os.getenv('SESSION_TIMEOUT', 3600))  # 1 hour
        self.max_failed_attempts = int(os.getenv('MAX_FAILED_LOGIN_ATTEMPTS', 5))
        self.lockout_duration = int(os.getenv('LOCKOUT_DURATION', 900))  # 15 minutes
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER,
        is_verified: bool = False
    ) -> User:
        """Create a new user."""
        try:
            with get_db_session() as session:
                # Check if user already exists
                existing_user = session.query(User).filter(
                    (User.username == username) | (User.email == email)
                ).first()
                
                if existing_user:
                    raise ProcessingError(
                        message="User with this username or email already exists",
                        error_code="USER_EXISTS",
                        severity=ErrorSeverity.MEDIUM
                    )
                
                # Create new user
                user = User(
                    id=secrets.token_urlsafe(16),
                    username=username,
                    email=email,
                    password_hash=self.hash_password(password),
                    role=role.value,
                    is_verified=is_verified
                )
                
                session.add(user)
                session.commit()
                
                # Log user creation
                audit_logger.log_operation(
                    operation="user_created",
                    resource_type="user",
                    resource_id=user.id,
                    new_values={
                        'username': username,
                        'email': email,
                        'role': role.value
                    }
                )
                
                self.logger.info(f"User created: {username} ({user.id})")
                return user
                
        except Exception as e:
            self.logger.error(f"Failed to create user {username}: {str(e)}")
            raise ProcessingError(
                message=f"Failed to create user: {str(e)}",
                error_code="USER_CREATION_FAILED",
                severity=ErrorSeverity.HIGH
            )
    
    def authenticate_user(self, username: str, password: str, ip_address: str = None) -> Optional[UserSession]:
        """Authenticate user and create session."""
        try:
            with get_db_session() as session:
                user = session.query(User).filter(User.username == username).first()
                
                if not user:
                    self.logger.warning(f"Authentication failed: user not found: {username}")
                    return None
                
                # Check if account is locked
                if user.locked_until and user.locked_until > datetime.utcnow():
                    self.logger.warning(f"Authentication failed: account locked: {username}")
                    return None
                
                # Check if account is active
                if not user.is_active:
                    self.logger.warning(f"Authentication failed: account inactive: {username}")
                    return None
                
                # Verify password
                if not self.verify_password(password, user.password_hash):
                    # Increment failed attempts
                    user.failed_login_attempts += 1
                    
                    # Lock account if too many failed attempts
                    if user.failed_login_attempts >= self.max_failed_attempts:
                        user.locked_until = datetime.utcnow() + timedelta(seconds=self.lockout_duration)
                        self.logger.warning(f"Account locked due to failed attempts: {username}")
                    
                    session.commit()
                    
                    # Log failed attempt
                    audit_logger.log_operation(
                        operation="login_failed",
                        resource_type="user",
                        resource_id=user.id,
                        ip_address=ip_address,
                        metadata={'reason': 'invalid_password'}
                    )
                    
                    return None
                
                # Reset failed attempts on successful login
                user.failed_login_attempts = 0
                user.locked_until = None
                user.last_login = datetime.utcnow()
                
                # Create session
                session_token = secrets.token_urlsafe(32)
                expires_at = datetime.utcnow() + timedelta(seconds=self.session_timeout)
                
                user_session = UserSession(
                    id=secrets.token_urlsafe(16),
                    user_id=user.id,
                    session_token=session_token,
                    expires_at=expires_at,
                    ip_address=ip_address
                )
                
                session.add(user_session)
                session.commit()
                
                # Create user session object
                permissions = ROLE_PERMISSIONS.get(UserRole(user.role), [])
                
                user_session_obj = UserSession(
                    user_id=user.id,
                    username=user.username,
                    role=UserRole(user.role),
                    permissions=permissions,
                    session_id=user_session.id,
                    created_at=user_session.created_at,
                    expires_at=user_session.expires_at,
                    ip_address=ip_address
                )
                
                # Log successful login
                audit_logger.log_operation(
                    operation="login_success",
                    resource_type="user",
                    resource_id=user.id,
                    ip_address=ip_address
                )
                
                self.logger.info(f"User authenticated: {username} ({user.id})")
                return user_session_obj
                
        except Exception as e:
            self.logger.error(f"Authentication error for {username}: {str(e)}")
            raise ProcessingError(
                message=f"Authentication failed: {str(e)}",
                error_code="AUTHENTICATION_ERROR",
                severity=ErrorSeverity.HIGH
            )
    
    def validate_session(self, session_token: str) -> Optional[UserSession]:
        """Validate session token and return user session."""
        try:
            with get_db_session() as session:
                user_session = session.query(UserSession).filter(
                    UserSession.session_token == session_token,
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.utcnow()
                ).first()
                
                if not user_session:
                    return None
                
                # Get user
                user = session.query(User).filter(User.id == user_session.user_id).first()
                if not user or not user.is_active:
                    return None
                
                # Update last activity
                user_session.expires_at = datetime.utcnow() + timedelta(seconds=self.session_timeout)
                session.commit()
                
                # Create user session object
                permissions = ROLE_PERMISSIONS.get(UserRole(user.role), [])
                
                return UserSession(
                    user_id=user.id,
                    username=user.username,
                    role=UserRole(user.role),
                    permissions=permissions,
                    session_id=user_session.id,
                    created_at=user_session.created_at,
                    expires_at=user_session.expires_at,
                    ip_address=user_session.ip_address,
                    user_agent=user_session.user_agent
                )
                
        except Exception as e:
            self.logger.error(f"Session validation error: {str(e)}")
            return None
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user by invalidating session."""
        try:
            with get_db_session() as session:
                user_session = session.query(UserSession).filter(
                    UserSession.session_token == session_token
                ).first()
                
                if user_session:
                    user_session.is_active = False
                    session.commit()
                    
                    # Log logout
                    audit_logger.log_operation(
                        operation="logout",
                        resource_type="user",
                        resource_id=user_session.user_id
                    )
                    
                    self.logger.info(f"User logged out: {user_session.user_id}")
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Logout error: {str(e)}")
            return False
    
    def create_api_key(
        self,
        user_id: str,
        key_name: str,
        permissions: List[Permission],
        expires_days: Optional[int] = None
    ) -> str:
        """Create API key for user."""
        try:
            with get_db_session() as session:
                # Generate API key
                api_key = secrets.token_urlsafe(32)
                key_hash = hashlib.sha256(api_key.encode()).hexdigest()
                
                # Set expiration
                expires_at = None
                if expires_days:
                    expires_at = datetime.utcnow() + timedelta(days=expires_days)
                
                # Create API key record
                api_key_record = APIKey(
                    id=secrets.token_urlsafe(16),
                    user_id=user_id,
                    key_name=key_name,
                    key_hash=key_hash,
                    permissions=','.join([p.value for p in permissions]),
                    expires_at=expires_at
                )
                
                session.add(api_key_record)
                session.commit()
                
                # Log API key creation
                audit_logger.log_operation(
                    operation="api_key_created",
                    resource_type="api_key",
                    resource_id=api_key_record.id,
                    user_id=user_id,
                    metadata={'key_name': key_name, 'permissions': [p.value for p in permissions]}
                )
                
                self.logger.info(f"API key created for user {user_id}: {key_name}")
                return api_key
                
        except Exception as e:
            self.logger.error(f"Failed to create API key: {str(e)}")
            raise ProcessingError(
                message=f"Failed to create API key: {str(e)}",
                error_code="API_KEY_CREATION_FAILED",
                severity=ErrorSeverity.MEDIUM
            )
    
    def list_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """List API keys for a user (metadata only; never returns the actual key)."""
        try:
            with get_db_session() as session:
                records = session.query(APIKey).filter(
                    APIKey.user_id == user_id
                ).order_by(APIKey.created_at.desc()).all()
                keys = []
                for r in records:
                    keys.append({
                        'id': r.id,
                        'key_name': r.key_name,
                        'created_at': r.created_at.isoformat() if r.created_at else None,
                        'expires_at': r.expires_at.isoformat() if r.expires_at else None,
                        'last_used': r.last_used.isoformat() if r.last_used else None,
                        'permissions': [p.strip() for p in (r.permissions or '').split(',') if p.strip()],
                        'is_active': r.is_active,
                    })
                return keys
        except Exception as e:
            self.logger.error(f"Failed to list API keys for user {user_id}: {str(e)}")
            return []

    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info."""
        try:
            with get_db_session() as session:
                key_hash = hashlib.sha256(api_key.encode()).hexdigest()
                
                api_key_record = session.query(APIKey).filter(
                    APIKey.key_hash == key_hash,
                    APIKey.is_active == True,
                    (APIKey.expires_at.is_(None)) | (APIKey.expires_at > datetime.utcnow())
                ).first()
                
                if not api_key_record:
                    return None
                
                # Get user
                user = session.query(User).filter(User.id == api_key_record.user_id).first()
                if not user or not user.is_active:
                    return None
                
                # Update last used
                api_key_record.last_used = datetime.utcnow()
                session.commit()
                
                # Parse permissions
                permissions = []
                if api_key_record.permissions:
                    permissions = [Permission(p) for p in api_key_record.permissions.split(',')]
                
                return {
                    'user_id': user.id,
                    'username': user.username,
                    'role': UserRole(user.role),
                    'permissions': permissions,
                    'api_key_id': api_key_record.id
                }
                
        except Exception as e:
            self.logger.error(f"API key validation error: {str(e)}")
            return None
    
    def has_permission(self, user_session: UserSession, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in user_session.permissions
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # This would be used with Flask decorators
                # Implementation depends on how it's integrated with the web app
                pass
            return wrapper
        return decorator


# Global authentication manager
auth_manager = AuthenticationManager()


def get_current_user(session_token: str) -> Optional[UserSession]:
    """Get current user from session token."""
    return auth_manager.validate_session(session_token)


def require_auth(permission: Optional[Permission] = None):
    """Decorator factory for authentication requirements."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This will be implemented in the web app integration
            pass
        return wrapper
    return decorator


def create_default_admin():
    """Create default admin user if none exists."""
    try:
        with get_db_session() as session:
            admin_exists = session.query(User).filter(User.role == UserRole.ADMIN.value).first()
            
            if not admin_exists:
                admin_user = auth_manager.create_user(
                    username=os.getenv('ADMIN_USERNAME', 'admin'),
                    email=os.getenv('ADMIN_EMAIL', 'admin@example.com'),
                    password=os.getenv('ADMIN_PASSWORD', 'admin123'),
                    role=UserRole.ADMIN,
                    is_verified=True
                )
                
                logger.info(f"Default admin user created: {admin_user.username}")
                return admin_user
            else:
                logger.info("Admin user already exists")
                return admin_exists
                
    except Exception as e:
        logger.error(f"Failed to create default admin: {str(e)}")
        raise
