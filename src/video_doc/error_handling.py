"""
Error handling, retry logic, and circuit breaker patterns for robust data processing.
"""

import time
import functools
import logging
from typing import Any, Callable, Optional, Dict, List, Type, Union
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager

from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
    before_sleep_log, after_log, retry_if_result, RetryError
)
import structlog

logger = structlog.get_logger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProcessingError(Exception):
    """Custom processing error with context."""
    message: str
    error_code: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context: Optional[Dict[str, Any]] = None
    retryable: bool = True
    original_exception: Optional[Exception] = None
    
    def __str__(self):
        return f"[{self.error_code}] {self.message}"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        
        self.logger = structlog.get_logger(__name__)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise ProcessingError(
                        "Circuit breaker is OPEN",
                        "CIRCUIT_BREAKER_OPEN",
                        ErrorSeverity.HIGH,
                        retryable=False
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise ProcessingError(
                    f"Circuit breaker failure: {str(e)}",
                    "CIRCUIT_BREAKER_FAILURE",
                    ErrorSeverity.MEDIUM,
                    original_exception=e
                )
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(
                "Circuit breaker opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [Exception]


def retry_with_config(config: RetryConfig):
    """Create retry decorator with custom configuration."""
    return retry(
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_exponential(
            multiplier=config.base_delay,
            max=config.max_delay,
            exp_base=config.exponential_base,
            jitter=config.jitter
        ),
        retry=retry_if_exception_type(tuple(config.retryable_exceptions)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )


def retry_on_failure(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: Optional[List[Type[Exception]]] = None
):
    """Decorator for retrying failed operations."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=retryable_exceptions
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        @retry_with_config(config)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    "Operation failed, retrying",
                    function=func.__name__,
                    error=str(e),
                    attempt=getattr(wrapper, 'retry_state', {}).get('attempt_number', 1)
                )
                raise
        
        return wrapper
    return decorator


class ErrorHandler:
    """Centralized error handling and recovery."""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> ProcessingError:
        """Handle and categorize errors."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Determine if error is retryable
        retryable = self._is_retryable_error(error)
        
        # Create processing error
        processing_error = ProcessingError(
            message=str(error),
            error_code=f"{error_type.upper()}_ERROR",
            severity=severity,
            context=context,
            retryable=retryable,
            original_exception=error
        )
        
        # Log error with context
        self.logger.error(
            "Error handled",
            error_type=error_type,
            error_message=str(error),
            severity=severity.value,
            retryable=retryable,
            context=context,
            error_count=self.error_counts[error_type]
        )
        
        return processing_error
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        retryable_errors = [
            ConnectionError,
            TimeoutError,
            OSError,
            IOError,
        ]
        
        # Check if it's a known retryable error type
        for retryable_type in retryable_errors:
            if isinstance(error, retryable_type):
                return True
        
        # Check error message for retryable patterns
        error_message = str(error).lower()
        retryable_patterns = [
            'timeout',
            'connection',
            'network',
            'temporary',
            'retry',
            'rate limit',
            'throttle'
        ]
        
        return any(pattern in error_message for pattern in retryable_patterns)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        total_errors = sum(self.error_counts.values())
        return {
            "total_errors": total_errors,
            "error_counts": self.error_counts,
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count
                }
                for name, cb in self.circuit_breakers.items()
            }
        }


# Global error handler instance
error_handler = ErrorHandler()


@contextmanager
def error_context(operation: str, **context):
    """Context manager for error handling with operation context."""
    logger.info(f"Starting operation: {operation}", **context)
    try:
        yield
        logger.info(f"Completed operation: {operation}", **context)
    except Exception as e:
        processing_error = error_handler.handle_error(
            e,
            context={"operation": operation, **context}
        )
        logger.error(
            f"Operation failed: {operation}",
            error=str(processing_error),
            **context
        )
        raise processing_error


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    error_context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        processing_error = error_handler.handle_error(
            e,
            context=error_context or {}
        )
        
        if not processing_error.retryable:
            logger.error(
                "Non-retryable error occurred",
                function=func.__name__,
                error=str(processing_error)
            )
            return default_return
        
        raise processing_error


# Decorators for common error handling patterns
def with_error_handling(
    operation_name: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    default_return: Any = None
):
    """Decorator to add error handling to functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                processing_error = error_handler.handle_error(
                    e,
                    context={"function": func.__name__, "operation": operation_name},
                    severity=severity
                )
                
                if not processing_error.retryable:
                    logger.error(
                        "Non-retryable error in function",
                        function=func.__name__,
                        operation=operation_name,
                        error=str(processing_error)
                    )
                    return default_return
                
                raise processing_error
        
        return wrapper
    return decorator


def with_circuit_breaker(service_name: str):
    """Decorator to add circuit breaker to functions."""
    def decorator(func: Callable) -> Callable:
        circuit_breaker = error_handler.get_circuit_breaker(service_name)
        return circuit_breaker(func)
    return decorator


def with_retry_and_circuit_breaker(
    service_name: str,
    retry_config: Optional[RetryConfig] = None
):
    """Decorator combining retry logic and circuit breaker."""
    def decorator(func: Callable) -> Callable:
        circuit_breaker = error_handler.get_circuit_breaker(service_name)
        
        if retry_config:
            retry_decorator = retry_with_config(retry_config)
            return retry_decorator(circuit_breaker(func))
        else:
            return circuit_breaker(func)
    
    return decorator


# Specific error handling for video processing components
class VideoProcessingError(ProcessingError):
    """Specific error for video processing operations."""
    pass


class TranscriptionError(ProcessingError):
    """Specific error for transcription operations."""
    pass


class DownloadError(ProcessingError):
    """Specific error for download operations."""
    pass


class ClassificationError(ProcessingError):
    """Specific error for classification operations."""
    pass


# Error recovery strategies
class RecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def can_recover(self, error: ProcessingError) -> bool:
        """Check if this strategy can recover from the error."""
        raise NotImplementedError
    
    def recover(self, error: ProcessingError, context: Dict[str, Any]) -> Any:
        """Attempt to recover from the error."""
        raise NotImplementedError


class FallbackRecovery(RecoveryStrategy):
    """Recovery strategy using fallback methods."""
    
    def __init__(self, fallback_func: Callable):
        self.fallback_func = fallback_func
    
    def can_recover(self, error: ProcessingError) -> bool:
        return error.retryable
    
    def recover(self, error: ProcessingError, context: Dict[str, Any]) -> Any:
        logger.info("Attempting fallback recovery", error=str(error))
        return self.fallback_func(**context)


class SkipRecovery(RecoveryStrategy):
    """Recovery strategy that skips the operation."""
    
    def can_recover(self, error: ProcessingError) -> bool:
        return True
    
    def recover(self, error: ProcessingError, context: Dict[str, Any]) -> Any:
        logger.warning("Skipping operation due to error", error=str(error))
        return None


# Recovery manager
class RecoveryManager:
    """Manages error recovery strategies."""
    
    def __init__(self):
        self.strategies: List[RecoveryStrategy] = []
    
    def add_strategy(self, strategy: RecoveryStrategy):
        """Add a recovery strategy."""
        self.strategies.append(strategy)
    
    def recover(self, error: ProcessingError, context: Dict[str, Any]) -> Any:
        """Attempt to recover from an error using available strategies."""
        for strategy in self.strategies:
            if strategy.can_recover(error):
                try:
                    return strategy.recover(error, context)
                except Exception as recovery_error:
                    logger.error(
                        "Recovery strategy failed",
                        strategy=strategy.__class__.__name__,
                        error=str(recovery_error)
                    )
                    continue
        
        logger.error("No recovery strategy succeeded", error=str(error))
        raise error


# Global recovery manager
recovery_manager = RecoveryManager()
