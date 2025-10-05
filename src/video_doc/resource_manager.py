"""
Resource management and monitoring system for Kubernetes deployment.
"""

import os
import time
import psutil
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

import redis
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ResourceMetrics:
    """Resource metrics data class."""
    cpu_percent: float
    memory_percent: float
    memory_used_bytes: int
    memory_total_bytes: int
    disk_usage_percent: float
    disk_used_bytes: int
    disk_total_bytes: int
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: datetime


@dataclass
class PodMetrics:
    """Pod metrics data class."""
    name: str
    namespace: str
    cpu_usage: float
    memory_usage: int
    memory_limit: int
    status: str
    restart_count: int
    age: str


class ResourceMonitor:
    """Monitor system resources and pod metrics."""
    
    def __init__(self, namespace: str = "video-processing"):
        self.namespace = namespace
        self.k8s_client = None
        self.redis_client = None
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()  # For in-cluster
        except:
            try:
                config.load_kube_config()  # For local development
            except Exception as e:
                logger.warning(f"Could not load Kubernetes config: {e}")
        
        self.k8s_client = client.CoreV1Api()
        self.apps_client = client.AppsV1Api()
        
        # Initialize Redis client
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
    
    def get_system_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_bytes = memory.used
            memory_total_bytes = memory.total
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_used_bytes = disk.used
            disk_total_bytes = disk.total
            
            # Network usage
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_bytes=memory_used_bytes,
                memory_total_bytes=memory_total_bytes,
                disk_usage_percent=disk_usage_percent,
                disk_used_bytes=disk_used_bytes,
                disk_total_bytes=disk_total_bytes,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            raise
    
    def get_pod_metrics(self) -> List[PodMetrics]:
        """Get metrics for all pods in the namespace."""
        pod_metrics = []
        
        try:
            # Get pods in namespace
            pods = self.k8s_client.list_namespaced_pod(namespace=self.namespace)
            
            for pod in pods.items:
                try:
                    # Get pod metrics
                    metrics = self.k8s_client.list_namespaced_pod(
                        namespace=self.namespace,
                        field_selector=f"metadata.name={pod.metadata.name}"
                    )
                    
                    if metrics.items:
                        pod_info = metrics.items[0]
                        
                        # Calculate age
                        age = datetime.now(timezone.utc) - pod_info.metadata.creation_timestamp.replace(tzinfo=timezone.utc)
                        age_str = str(age).split('.')[0]  # Remove microseconds
                        
                        # Get resource usage from pod status
                        cpu_usage = 0.0
                        memory_usage = 0
                        memory_limit = 0
                        
                        if pod_info.status.container_statuses:
                            for container in pod_info.status.container_statuses:
                                if container.resources and container.resources.limits:
                                    if 'memory' in container.resources.limits:
                                        memory_limit += self._parse_memory_limit(container.resources.limits['memory'])
                        
                        pod_metrics.append(PodMetrics(
                            name=pod.metadata.name,
                            namespace=pod.metadata.namespace,
                            cpu_usage=cpu_usage,
                            memory_usage=memory_usage,
                            memory_limit=memory_limit,
                            status=pod.status.phase,
                            restart_count=sum(cs.restart_count for cs in pod.status.container_statuses or []),
                            age=age_str
                        ))
                        
                except Exception as e:
                    logger.warning(f"Failed to get metrics for pod {pod.metadata.name}: {e}")
                    continue
            
        except ApiException as e:
            logger.error(f"Kubernetes API error: {e}")
        except Exception as e:
            logger.error(f"Failed to get pod metrics: {e}")
        
        return pod_metrics
    
    def _parse_memory_limit(self, memory_str: str) -> int:
        """Parse memory limit string to bytes."""
        try:
            if memory_str.endswith('Gi'):
                return int(float(memory_str[:-2]) * 1024**3)
            elif memory_str.endswith('Mi'):
                return int(float(memory_str[:-2]) * 1024**2)
            elif memory_str.endswith('Ki'):
                return int(float(memory_str[:-2]) * 1024)
            else:
                return int(memory_str)
        except:
            return 0
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get status of all deployments."""
        try:
            deployments = self.apps_client.list_namespaced_deployment(namespace=self.namespace)
            
            deployment_status = {}
            for deployment in deployments.items:
                deployment_status[deployment.metadata.name] = {
                    "replicas": deployment.spec.replicas,
                    "ready_replicas": deployment.status.ready_replicas or 0,
                    "available_replicas": deployment.status.available_replicas or 0,
                    "unavailable_replicas": deployment.status.unavailable_replicas or 0,
                    "updated_replicas": deployment.status.updated_replicas or 0,
                    "conditions": [
                        {
                            "type": condition.type,
                            "status": condition.status,
                            "message": condition.message
                        }
                        for condition in deployment.status.conditions or []
                    ]
                }
            
            return deployment_status
            
        except ApiException as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {}
    
    def get_hpa_status(self) -> Dict[str, Any]:
        """Get status of Horizontal Pod Autoscalers."""
        try:
            # Note: This requires the autoscaling/v2 API
            from kubernetes.client import CustomObjectsApi
            custom_api = CustomObjectsApi()
            
            hpa_list = custom_api.list_namespaced_custom_object(
                group="autoscaling",
                version="v2",
                namespace=self.namespace,
                plural="horizontalpodautoscalers"
            )
            
            hpa_status = {}
            for hpa in hpa_list.get('items', []):
                metadata = hpa['metadata']
                spec = hpa['spec']
                status = hpa['status']
                
                hpa_status[metadata['name']] = {
                    "min_replicas": spec['minReplicas'],
                    "max_replicas": spec['maxReplicas'],
                    "current_replicas": status.get('currentReplicas', 0),
                    "desired_replicas": status.get('desiredReplicas', 0),
                    "current_cpu_utilization": status.get('currentCPUUtilizationPercentage'),
                    "conditions": status.get('conditions', [])
                }
            
            return hpa_status
            
        except Exception as e:
            logger.warning(f"Could not get HPA status: {e}")
            return {}
    
    def scale_deployment(self, deployment_name: str, replicas: int) -> bool:
        """Scale a deployment to specified number of replicas."""
        try:
            # Get current deployment
            deployment = self.apps_client.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Update replicas
            deployment.spec.replicas = replicas
            
            # Apply update
            self.apps_client.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"Scaled deployment {deployment_name} to {replicas} replicas")
            return True
            
        except ApiException as e:
            logger.error(f"Failed to scale deployment {deployment_name}: {e}")
            return False
    
    def start_monitoring(self, interval: int = 30):
        """Start continuous resource monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started resource monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped resource monitoring")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                system_metrics = self.get_system_metrics()
                pod_metrics = self.get_pod_metrics()
                deployment_status = self.get_deployment_status()
                hpa_status = self.get_hpa_status()
                
                # Store metrics in Redis
                if self.redis_client:
                    self._store_metrics_in_redis({
                        "system": system_metrics.__dict__,
                        "pods": [pod.__dict__ for pod in pod_metrics],
                        "deployments": deployment_status,
                        "hpa": hpa_status,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
                # Log metrics
                logger.info(
                    "Resource metrics collected",
                    cpu_percent=system_metrics.cpu_percent,
                    memory_percent=system_metrics.memory_percent,
                    disk_percent=system_metrics.disk_usage_percent,
                    pod_count=len(pod_metrics),
                    deployment_count=len(deployment_status)
                )
                
                # Check for scaling conditions
                self._check_scaling_conditions(system_metrics, pod_metrics, deployment_status)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(interval)
    
    def _store_metrics_in_redis(self, metrics: Dict[str, Any]):
        """Store metrics in Redis for external monitoring."""
        try:
            key = f"video_processing:metrics:{int(time.time())}"
            self.redis_client.setex(key, 3600, str(metrics))  # Expire in 1 hour
            
            # Also store latest metrics
            self.redis_client.set("video_processing:metrics:latest", str(metrics))
            
        except Exception as e:
            logger.error(f"Failed to store metrics in Redis: {e}")
    
    def _check_scaling_conditions(self, system_metrics: ResourceMetrics, pod_metrics: List[PodMetrics], deployment_status: Dict[str, Any]):
        """Check if scaling is needed based on current metrics."""
        try:
            # Check CPU usage
            if system_metrics.cpu_percent > 80:
                logger.warning(f"High CPU usage detected: {system_metrics.cpu_percent}%")
                # Could trigger scaling here
            
            # Check memory usage
            if system_metrics.memory_percent > 85:
                logger.warning(f"High memory usage detected: {system_metrics.memory_percent}%")
                # Could trigger scaling here
            
            # Check pod health
            unhealthy_pods = [pod for pod in pod_metrics if pod.status != "Running"]
            if unhealthy_pods:
                logger.warning(f"Unhealthy pods detected: {len(unhealthy_pods)}")
            
            # Check deployment readiness
            for deployment_name, status in deployment_status.items():
                if status["ready_replicas"] < status["replicas"]:
                    logger.warning(f"Deployment {deployment_name} not fully ready: {status['ready_replicas']}/{status['replicas']}")
            
        except Exception as e:
            logger.error(f"Error checking scaling conditions: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        try:
            system_metrics = self.get_system_metrics()
            pod_metrics = self.get_pod_metrics()
            deployment_status = self.get_deployment_status()
            hpa_status = self.get_hpa_status()
            
            return {
                "system": {
                    "cpu_percent": system_metrics.cpu_percent,
                    "memory_percent": system_metrics.memory_percent,
                    "disk_percent": system_metrics.disk_usage_percent,
                    "timestamp": system_metrics.timestamp.isoformat()
                },
                "pods": {
                    "total": len(pod_metrics),
                    "running": len([p for p in pod_metrics if p.status == "Running"]),
                    "failed": len([p for p in pod_metrics if p.status == "Failed"]),
                    "pending": len([p for p in pod_metrics if p.status == "Pending"])
                },
                "deployments": {
                    "total": len(deployment_status),
                    "ready": len([d for d in deployment_status.values() if d["ready_replicas"] == d["replicas"]]),
                    "scaling": len([d for d in deployment_status.values() if d["ready_replicas"] != d["replicas"]])
                },
                "hpa": {
                    "total": len(hpa_status),
                    "active": len([h for h in hpa_status.values() if h["current_replicas"] > 0])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {"error": str(e)}


class ResourceManager:
    """Manage resource allocation and scaling decisions."""
    
    def __init__(self, namespace: str = "video-processing"):
        self.namespace = namespace
        self.monitor = ResourceMonitor(namespace)
        self.scaling_enabled = True
        self.scaling_thresholds = {
            "cpu_scale_up": 75.0,
            "cpu_scale_down": 25.0,
            "memory_scale_up": 80.0,
            "memory_scale_down": 30.0,
            "queue_length_scale_up": 10,
            "queue_length_scale_down": 2
        }
    
    def enable_auto_scaling(self):
        """Enable automatic scaling."""
        self.scaling_enabled = True
        logger.info("Auto-scaling enabled")
    
    def disable_auto_scaling(self):
        """Disable automatic scaling."""
        self.scaling_enabled = False
        logger.info("Auto-scaling disabled")
    
    def update_scaling_thresholds(self, thresholds: Dict[str, float]):
        """Update scaling thresholds."""
        self.scaling_thresholds.update(thresholds)
        logger.info("Scaling thresholds updated", thresholds=thresholds)
    
    def should_scale_up(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if scaling up is needed."""
        if not self.scaling_enabled:
            return False, "Auto-scaling disabled"
        
        system = metrics.get("system", {})
        pods = metrics.get("pods", {})
        
        # Check CPU threshold
        if system.get("cpu_percent", 0) > self.scaling_thresholds["cpu_scale_up"]:
            return True, f"High CPU usage: {system['cpu_percent']}%"
        
        # Check memory threshold
        if system.get("memory_percent", 0) > self.scaling_thresholds["memory_scale_up"]:
            return True, f"High memory usage: {system['memory_percent']}%"
        
        # Check pod availability
        if pods.get("running", 0) < pods.get("total", 0):
            return True, f"Unhealthy pods: {pods['total'] - pods['running']}"
        
        return False, "No scaling needed"
    
    def should_scale_down(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if scaling down is needed."""
        if not self.scaling_enabled:
            return False, "Auto-scaling disabled"
        
        system = metrics.get("system", {})
        
        # Check CPU threshold
        if system.get("cpu_percent", 0) < self.scaling_thresholds["cpu_scale_down"]:
            return True, f"Low CPU usage: {system['cpu_percent']}%"
        
        # Check memory threshold
        if system.get("memory_percent", 0) < self.scaling_thresholds["memory_scale_down"]:
            return True, f"Low memory usage: {system['memory_percent']}%"
        
        return False, "No scaling needed"
    
    def execute_scaling_action(self, deployment_name: str, action: str, replicas: int) -> bool:
        """Execute a scaling action."""
        try:
            if action == "scale_up":
                logger.info(f"Scaling up {deployment_name} to {replicas} replicas")
            elif action == "scale_down":
                logger.info(f"Scaling down {deployment_name} to {replicas} replicas")
            
            success = self.monitor.scale_deployment(deployment_name, replicas)
            
            if success:
                logger.info(f"Successfully {action} {deployment_name}")
            else:
                logger.error(f"Failed to {action} {deployment_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing scaling action: {e}")
            return False


# Global resource manager instance
resource_manager = ResourceManager()


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    return resource_manager
