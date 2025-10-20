"""
Enhanced security features for the authentication system.
"""

import os
import time
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from .auth import auth_manager, User, UserSession
from .monitoring import get_logger, audit_logger
from .error_handling import ProcessingError, ErrorSeverity

logger = get_logger(__name__)


class SecurityEvent(str, Enum):
    """Security event types."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    ACCOUNT_LOCKED = "account_locked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    API_KEY_CREATED = "api_key_created"
    API_KEY_USED = "api_key_used"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class SecurityAlert:
    """Security alert data."""
    alert_id: str
    user_id: Optional[str]
    event_type: SecurityEvent
    severity: str  # low, medium, high, critical
    message: str
    ip_address: str
    user_agent: str
    timestamp: datetime
    resolved: bool = False
    metadata: Optional[Dict[str, Any]] = None


class SecurityManager:
    """Enhanced security manager with advanced features."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.security_alerts: List[SecurityAlert] = []
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.suspicious_ips: Dict[str, int] = {}
        self.rate_limits: Dict[str, List[datetime]] = {}
        
        # Security configuration
        self.max_failed_attempts_per_ip = int(os.getenv('MAX_FAILED_ATTEMPTS_PER_IP', 10))
        self.rate_limit_window = int(os.getenv('RATE_LIMIT_WINDOW', 300))  # 5 minutes
        self.max_requests_per_window = int(os.getenv('MAX_REQUESTS_PER_WINDOW', 100))
        self.suspicious_threshold = int(os.getenv('SUSPICIOUS_THRESHOLD', 5))
    
    def check_rate_limit(self, ip_address: str) -> bool:
        """Check if IP is within rate limits."""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.rate_limit_window)
        
        if ip_address not in self.rate_limits:
            self.rate_limits[ip_address] = []
        
        # Clean old requests
        self.rate_limits[ip_address] = [
            req_time for req_time in self.rate_limits[ip_address]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.rate_limits[ip_address]) >= self.max_requests_per_window:
            self._create_security_alert(
                user_id=None,
                event_type=SecurityEvent.SUSPICIOUS_ACTIVITY,
                severity="high",
                message=f"Rate limit exceeded for IP {ip_address}",
                ip_address=ip_address,
                user_agent="",
                metadata={"requests_count": len(self.rate_limits[ip_address])}
            )
            return False
        
        # Add current request
        self.rate_limits[ip_address].append(now)
        return True
    
    def check_suspicious_activity(self, ip_address: str, user_id: Optional[str] = None) -> bool:
        """Check for suspicious activity patterns."""
        # Check failed attempts from IP
        failed_count = len(self.failed_attempts.get(ip_address, []))
        
        if failed_count >= self.suspicious_threshold:
            self._create_security_alert(
                user_id=user_id,
                event_type=SecurityEvent.SUSPICIOUS_ACTIVITY,
                severity="high",
                message=f"Suspicious activity detected from IP {ip_address}",
                ip_address=ip_address,
                user_agent="",
                metadata={"failed_attempts": failed_count}
            )
            return True
        
        return False
    
    def record_failed_attempt(self, ip_address: str, username: Optional[str] = None):
        """Record a failed login attempt."""
        now = datetime.now(timezone.utc)
        
        if ip_address not in self.failed_attempts:
            self.failed_attempts[ip_address] = []
        
        self.failed_attempts[ip_address].append(now)
        
        # Clean old attempts (older than 1 hour)
        cutoff = now - timedelta(hours=1)
        self.failed_attempts[ip_address] = [
            attempt for attempt in self.failed_attempts[ip_address]
            if attempt > cutoff
        ]
        
        # Log failed attempt
        audit_logger.log_operation(
            operation="login_failed",
            resource_type="user",
            resource_id=username,
            ip_address=ip_address,
            metadata={"failed_attempts": len(self.failed_attempts[ip_address])}
        )
    
    def _create_security_alert(
        self,
        user_id: Optional[str],
        event_type: SecurityEvent,
        severity: str,
        message: str,
        ip_address: str,
        user_agent: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create a security alert."""
        alert = SecurityAlert(
            alert_id=secrets.token_urlsafe(16),
            user_id=user_id,
            event_type=event_type,
            severity=severity,
            message=message,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata
        )
        
        self.security_alerts.append(alert)
        
        # Log security alert
        self.logger.warning(
            f"Security alert: {message}",
            alert_id=alert.alert_id,
            event_type=event_type.value,
            severity=severity,
            ip_address=ip_address,
            user_id=user_id,
            metadata=metadata
        )
        
        # Send alert to monitoring system (if configured)
        self._send_security_notification(alert)
    
    def _send_security_notification(self, alert: SecurityAlert):
        """Send security notification (placeholder for integration)."""
        # This could integrate with:
        # - Email notifications
        # - Slack/Discord webhooks
        # - SMS alerts
        # - Security monitoring systems
        
        if alert.severity in ["high", "critical"]:
            self.logger.critical(f"CRITICAL SECURITY ALERT: {alert.message}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary for dashboard."""
        now = datetime.now(timezone.utc)
        last_24h = now - timedelta(hours=24)
        
        recent_alerts = [
            alert for alert in self.security_alerts
            if alert.timestamp > last_24h
        ]
        
        return {
            "total_alerts": len(self.security_alerts),
            "recent_alerts": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a.severity == "critical"]),
            "high_alerts": len([a for a in recent_alerts if a.severity == "high"]),
            "suspicious_ips": len(self.suspicious_ips),
            "failed_attempts": sum(len(attempts) for attempts in self.failed_attempts.values()),
            "rate_limited_ips": len([
                ip for ip, requests in self.rate_limits.items()
                if len(requests) >= self.max_requests_per_window
            ])
        }


class PasswordPolicy:
    """Password policy enforcement."""
    
    def __init__(self):
        self.min_length = int(os.getenv('PASSWORD_MIN_LENGTH', 8))
        self.require_uppercase = os.getenv('PASSWORD_REQUIRE_UPPERCASE', 'true').lower() == 'true'
        self.require_lowercase = os.getenv('PASSWORD_REQUIRE_LOWERCASE', 'true').lower() == 'true'
        self.require_numbers = os.getenv('PASSWORD_REQUIRE_NUMBERS', 'true').lower() == 'true'
        self.require_special = os.getenv('PASSWORD_REQUIRE_SPECIAL', 'true').lower() == 'true'
        self.forbidden_passwords = self._load_forbidden_passwords()
    
    def _load_forbidden_passwords(self) -> List[str]:
        """Load list of forbidden passwords."""
        # Common weak passwords
        return [
            "password", "123456", "123456789", "qwerty", "abc123",
            "password123", "admin", "letmein", "welcome", "monkey",
            "1234567890", "password1", "qwerty123", "dragon", "master"
        ]
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password against policy."""
        errors = []
        warnings = []
        
        # Length check
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")
        
        # Character requirements
        if self.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if self.require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        # Forbidden passwords
        if password.lower() in self.forbidden_passwords:
            errors.append("Password is too common and not allowed")
        
        # Additional checks
        if len(set(password)) < len(password) * 0.5:  # Less than 50% unique characters
            warnings.append("Password has many repeated characters")
        
        if password.isdigit() or password.isalpha():
            warnings.append("Password should contain both letters and numbers")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "strength": self._calculate_strength(password)
        }
    
    def _calculate_strength(self, password: str) -> str:
        """Calculate password strength."""
        score = 0
        
        # Length bonus
        if len(password) >= 8:
            score += 1
        if len(password) >= 12:
            score += 1
        if len(password) >= 16:
            score += 1
        
        # Character variety
        if any(c.isupper() for c in password):
            score += 1
        if any(c.islower() for c in password):
            score += 1
        if any(c.isdigit() for c in password):
            score += 1
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1
        
        # Uniqueness
        if len(set(password)) >= len(password) * 0.7:
            score += 1
        
        if score <= 2:
            return "weak"
        elif score <= 4:
            return "medium"
        elif score <= 6:
            return "strong"
        else:
            return "very_strong"


class TwoFactorAuth:
    """Two-factor authentication support."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.enabled = os.getenv('TWO_FACTOR_ENABLED', 'false').lower() == 'true'
        self.issuer = os.getenv('TWO_FACTOR_ISSUER', 'Video Processing System')
    
    def generate_secret(self, user_id: str) -> str:
        """Generate TOTP secret for user."""
        if not self.enabled:
            return ""
        
        # This would integrate with pyotp library
        # For now, return a placeholder
        return f"TOTP_SECRET_FOR_{user_id}"
    
    def generate_qr_code(self, user_id: str, secret: str) -> str:
        """Generate QR code for TOTP setup."""
        if not self.enabled:
            return ""
        
        # This would generate actual QR code
        return f"otpauth://totp/{self.issuer}:{user_id}?secret={secret}&issuer={self.issuer}"
    
    def verify_token(self, secret: str, token: str) -> bool:
        """Verify TOTP token."""
        if not self.enabled:
            return True
        
        # This would verify with pyotp
        # For now, return True for demo
        return True


# Global security instances
security_manager = SecurityManager()
password_policy = PasswordPolicy()
two_factor_auth = TwoFactorAuth()
