#!/usr/bin/env python3
"""
Command-line health check utility for video processing services.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any

from src.video_doc.health_checks import get_health_status, get_health_summary, get_service_health


def print_health_status(data: Dict[str, Any], detailed: bool = False):
    """Print health status in a formatted way."""
    
    # Overall status with color coding
    status = data['status'].upper()
    status_colors = {
        'HEALTHY': '\033[92m',    # Green
        'DEGRADED': '\033[93m',    # Yellow
        'UNHEALTHY': '\033[91m',   # Red
        'UNKNOWN': '\033[90m'      # Gray
    }
    reset_color = '\033[0m'
    
    color = status_colors.get(status, '')
    print(f"\n{color}=== SYSTEM HEALTH: {status} ==={reset_color}")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Uptime: {format_uptime(data['uptime_seconds'])}")
    print(f"Version: {data.get('version', 'Unknown')}")
    print(f"Active Jobs: {data['active_jobs']}")
    print(f"Queue Size: {data['queue_size']}")
    
    if detailed:
        print("\n=== SYSTEM RESOURCES ===")
        resources = data.get('system_resources', {})
        if 'cpu_percent' in resources:
            print(f"CPU Usage: {resources['cpu_percent']:.1f}%")
        if 'memory_percent' in resources:
            print(f"Memory Usage: {resources['memory_percent']:.1f}%")
        
        print("\n=== SERVICE STATUS ===")
        services = data.get('services', {})
        for service_name, service in services.items():
            service_color = status_colors.get(service['status'].upper(), '')
            print(f"\n{service_color}● {service_name}: {service['status'].upper()}{reset_color}")
            
            if service.get('error_message'):
                print(f"  Error: {service['error_message']}")
            
            if service.get('warnings'):
                print(f"  Warnings:")
                for warning in service['warnings']:
                    print(f"    - {warning}")
            
            if service.get('details') and detailed:
                print(f"  Details:")
                for key, value in service['details'].items():
                    print(f"    {key}: {format_value(value)}")


def print_service_status(service_name: str, service_data: Dict[str, Any]):
    """Print status for a specific service."""
    if service_data is None:
        print(f"Service '{service_name}' not found")
        return
    
    status_colors = {
        'HEALTHY': '\033[92m',
        'DEGRADED': '\033[93m',
        'UNHEALTHY': '\033[91m',
        'UNKNOWN': '\033[90m'
    }
    reset_color = '\033[0m'
    
    status = service_data['status'].upper()
    color = status_colors.get(status, '')
    
    print(f"\n{color}=== {service_name.upper()} SERVICE STATUS ==={reset_color}")
    print(f"Status: {status}")
    print(f"Timestamp: {service_data['timestamp']}")
    
    if service_data.get('error_message'):
        print(f"Error: {service_data['error_message']}")
    
    if service_data.get('warnings'):
        print(f"Warnings:")
        for warning in service_data['warnings']:
            print(f"  - {warning}")
    
    if service_data.get('details'):
        print(f"Details:")
        for key, value in service_data['details'].items():
            print(f"  {key}: {format_value(value)}")


def format_uptime(seconds: float) -> str:
    """Format uptime in a human-readable way."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def format_value(value: Any) -> str:
    """Format a value for display."""
    if isinstance(value, dict):
        return json.dumps(value, indent=2)
    elif isinstance(value, (list, tuple)):
        return ', '.join(str(v) for v in value)
    else:
        return str(value)


def watch_mode(interval: int = 30):
    """Watch mode - continuously monitor health status."""
    print("Starting health monitoring (Press Ctrl+C to stop)...")
    print(f"Refresh interval: {interval} seconds\n")
    
    try:
        while True:
            # Clear screen (works on most terminals)
            print('\033[2J\033[H', end='')
            
            try:
                data = get_health_summary()
                print_health_status(data, detailed=False)
                
                # Show timestamp
                print(f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Next update in: {interval} seconds")
                
            except Exception as e:
                print(f"Error fetching health status: {e}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nHealth monitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description='Video Processing Health Check Utility')
    parser.add_argument('--detailed', '-d', action='store_true', 
                       help='Show detailed health information')
    parser.add_argument('--service', '-s', type=str, 
                       help='Check specific service health')
    parser.add_argument('--watch', '-w', action='store_true', 
                       help='Watch mode - continuously monitor health')
    parser.add_argument('--interval', '-i', type=int, default=30,
                       help='Watch mode refresh interval in seconds (default: 30)')
    parser.add_argument('--json', '-j', action='store_true',
                       help='Output in JSON format')
    
    args = parser.parse_args()
    
    try:
        if args.service:
            # Check specific service
            service_data = get_service_health(args.service)
            if args.json:
                print(json.dumps(service_data, indent=2))
            else:
                print_service_status(args.service, service_data)
        
        elif args.watch:
            # Watch mode
            watch_mode(args.interval)
        
        else:
            # Standard health check
            if args.detailed:
                data = get_health_status()
            else:
                data = get_health_summary()
            
            if args.json:
                print(json.dumps(data, indent=2))
            else:
                print_health_status(data, detailed=args.detailed)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
