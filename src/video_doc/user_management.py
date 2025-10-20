"""
Enhanced user management features.
"""

import os
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum

from .auth import auth_manager, User, UserRole, Permission
from .monitoring import get_logger, audit_logger
from .error_handling import ProcessingError, ErrorSeverity
from .security_enhancements import security_manager, password_policy

logger = get_logger(__name__)


class UserStatus(str, Enum):
    """User status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


@dataclass
class UserProfile:
    """Extended user profile information."""
    user_id: str
    username: str
    email: str
    role: UserRole
    status: UserStatus
    created_at: datetime
    last_login: Optional[datetime]
    login_count: int
    failed_login_attempts: int
    locked_until: Optional[datetime]
    preferences: Dict[str, Any]
    api_keys_count: int
    active_sessions: int
    total_jobs: int
    completed_jobs: int
    failed_jobs: int


class UserManager:
    """Enhanced user management system."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get comprehensive user profile."""
        try:
            from .database import get_db_session
            
            with get_db_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return None
                
                # Get additional statistics
                from .database import ProcessingJob, ProcessingStatus
                
                total_jobs = session.query(ProcessingJob).filter(
                    ProcessingJob.user_id == user_id
                ).count()
                
                completed_jobs = session.query(ProcessingJob).filter(
                    ProcessingJob.user_id == user_id,
                    ProcessingJob.status == ProcessingStatus.COMPLETED
                ).count()
                
                failed_jobs = session.query(ProcessingJob).filter(
                    ProcessingJob.user_id == user_id,
                    ProcessingJob.status == ProcessingStatus.FAILED
                ).count()
                
                # Count active sessions
                from .auth import UserSession as UserSessionModel
                active_sessions = session.query(UserSessionModel).filter(
                    UserSessionModel.user_id == user_id,
                    UserSessionModel.is_active == True,
                    UserSessionModel.expires_at > datetime.utcnow()
                ).count()
                
                # Count API keys
                from .auth import APIKey
                api_keys_count = session.query(APIKey).filter(
                    APIKey.user_id == user_id,
                    APIKey.is_active == True
                ).count()
                
                return UserProfile(
                    user_id=user.id,
                    username=user.username,
                    email=user.email,
                    role=UserRole(user.role),
                    status=UserStatus.ACTIVE if user.is_active else UserStatus.INACTIVE,
                    created_at=user.created_at,
                    last_login=user.last_login,
                    login_count=0,  # Would need to track this
                    failed_login_attempts=user.failed_login_attempts,
                    locked_until=user.locked_until,
                    preferences=json.loads(user.preferences) if user.preferences else {},
                    api_keys_count=api_keys_count,
                    active_sessions=active_sessions,
                    total_jobs=total_jobs,
                    completed_jobs=completed_jobs,
                    failed_jobs=failed_jobs
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get user profile for {user_id}: {str(e)}")
            return None
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        try:
            from .database import get_db_session
            
            with get_db_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return False
                
                user.preferences = json.dumps(preferences)
                user.updated_at = datetime.utcnow()
                session.commit()
                
                # Log preference update
                audit_logger.log_operation(
                    operation="preferences_updated",
                    resource_type="user",
                    resource_id=user_id,
                    user_id=user_id,
                    new_values={"preferences": preferences}
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update preferences for {user_id}: {str(e)}")
            return False
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> Dict[str, Any]:
        """Change user password with validation."""
        try:
            from .database import get_db_session
            
            with get_db_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return {"success": False, "error": "User not found"}
                
                # Verify old password
                if not auth_manager.verify_password(old_password, user.password_hash):
                    return {"success": False, "error": "Current password is incorrect"}
                
                # Validate new password
                validation = password_policy.validate_password(new_password)
                if not validation["valid"]:
                    return {
                        "success": False,
                        "error": "Password does not meet requirements",
                        "details": validation["errors"]
                    }
                
                # Update password
                user.password_hash = auth_manager.hash_password(new_password)
                user.updated_at = datetime.utcnow()
                session.commit()
                
                # Log password change
                audit_logger.log_operation(
                    operation="password_changed",
                    resource_type="user",
                    resource_id=user_id,
                    user_id=user_id
                )
                
                return {
                    "success": True,
                    "message": "Password changed successfully",
                    "strength": validation["strength"],
                    "warnings": validation["warnings"]
                }
                
        except Exception as e:
            self.logger.error(f"Failed to change password for {user_id}: {str(e)}")
            return {"success": False, "error": "Password change failed"}
    
    def suspend_user(self, user_id: str, reason: str, admin_user_id: str) -> bool:
        """Suspend user account."""
        try:
            from .database import get_db_session
            
            with get_db_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return False
                
                user.is_active = False
                user.updated_at = datetime.utcnow()
                session.commit()
                
                # Log suspension
                audit_logger.log_operation(
                    operation="user_suspended",
                    resource_type="user",
                    resource_id=user_id,
                    user_id=admin_user_id,
                    metadata={"reason": reason}
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to suspend user {user_id}: {str(e)}")
            return False
    
    def activate_user(self, user_id: str, admin_user_id: str) -> bool:
        """Activate user account."""
        try:
            from .database import get_db_session
            
            with get_db_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return False
                
                user.is_active = True
                user.updated_at = datetime.utcnow()
                session.commit()
                
                # Log activation
                audit_logger.log_operation(
                    operation="user_activated",
                    resource_type="user",
                    resource_id=user_id,
                    user_id=admin_user_id
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to activate user {user_id}: {str(e)}")
            return False
    
    def get_user_statistics(self) -> Dict[str, Any]:
        """Get user statistics for admin dashboard."""
        try:
            from .database import get_db_session, User as UserModel
            
            with get_db_session() as session:
                total_users = session.query(UserModel).count()
                active_users = session.query(UserModel).filter(UserModel.is_active == True).count()
                admin_users = session.query(UserModel).filter(UserModel.role == UserRole.ADMIN.value).count()
                regular_users = session.query(UserModel).filter(UserModel.role == UserRole.USER.value).count()
                
                # Users created in last 30 days
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                recent_users = session.query(UserModel).filter(
                    UserModel.created_at > thirty_days_ago
                ).count()
                
                return {
                    "total_users": total_users,
                    "active_users": active_users,
                    "inactive_users": total_users - active_users,
                    "admin_users": admin_users,
                    "regular_users": regular_users,
                    "recent_users": recent_users,
                    "user_growth_rate": (recent_users / total_users * 100) if total_users > 0 else 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get user statistics: {str(e)}")
            return {}


class SessionManager:
    """Enhanced session management."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all active sessions for user."""
        try:
            from .database import get_db_session
            from .auth import UserSession as UserSessionModel
            
            with get_db_session() as session:
                user_sessions = session.query(UserSessionModel).filter(
                    UserSessionModel.user_id == user_id,
                    UserSessionModel.is_active == True
                ).all()
                
                sessions = []
                for user_session in user_sessions:
                    sessions.append({
                        "session_id": user_session.id,
                        "created_at": user_session.created_at,
                        "expires_at": user_session.expires_at,
                        "ip_address": user_session.ip_address,
                        "user_agent": user_session.user_agent,
                        "is_current": False  # Would need to compare with current session
                    })
                
                return sessions
                
        except Exception as e:
            self.logger.error(f"Failed to get sessions for {user_id}: {str(e)}")
            return []
    
    def revoke_session(self, session_id: str, admin_user_id: str) -> bool:
        """Revoke a specific session."""
        try:
            from .database import get_db_session
            from .auth import UserSession as UserSessionModel
            
            with get_db_session() as session:
                user_session = session.query(UserSessionModel).filter(
                    UserSessionModel.id == session_id
                ).first()
                
                if user_session:
                    user_session.is_active = False
                    session.commit()
                    
                    # Log session revocation
                    audit_logger.log_operation(
                        operation="session_revoked",
                        resource_type="session",
                        resource_id=session_id,
                        user_id=admin_user_id,
                        metadata={"target_user_id": user_session.user_id}
                    )
                    
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to revoke session {session_id}: {str(e)}")
            return False
    
    def revoke_all_sessions(self, user_id: str, admin_user_id: str) -> int:
        """Revoke all sessions for a user."""
        try:
            from .database import get_db_session
            from .auth import UserSession as UserSessionModel
            
            with get_db_session() as session:
                user_sessions = session.query(UserSessionModel).filter(
                    UserSessionModel.user_id == user_id,
                    UserSessionModel.is_active == True
                ).all()
                
                revoked_count = 0
                for user_session in user_sessions:
                    user_session.is_active = False
                    revoked_count += 1
                
                session.commit()
                
                # Log bulk session revocation
                audit_logger.log_operation(
                    operation="all_sessions_revoked",
                    resource_type="user",
                    resource_id=user_id,
                    user_id=admin_user_id,
                    metadata={"revoked_count": revoked_count}
                )
                
                return revoked_count
                
        except Exception as e:
            self.logger.error(f"Failed to revoke all sessions for {user_id}: {str(e)}")
            return 0


# Global user management instances
user_manager = UserManager()
session_manager = SessionManager()
