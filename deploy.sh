#!/bin/bash

# Kubernetes deployment script for video processing application

set -e

# Configuration
NAMESPACE="video-processing"
IMAGE_NAME="video-processing"
IMAGE_TAG="latest"
REGISTRY="your-registry.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    log_info "kubectl is available"
}

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    log_info "Docker is available"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Push image to registry (optional)
push_image() {
    if [ "$1" = "--push" ]; then
        log_info "Pushing image to registry..."
        docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
        docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
        
        if [ $? -eq 0 ]; then
            log_info "Image pushed successfully"
        else
            log_error "Failed to push image"
            exit 1
        fi
    fi
}

# Create namespace
create_namespace() {
    log_info "Creating namespace ${NAMESPACE}..."
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    log_info "Namespace ${NAMESPACE} created or already exists"
}

# Deploy PostgreSQL
deploy_postgres() {
    log_info "Deploying PostgreSQL..."
    kubectl apply -f k8s/deployment.yaml -n ${NAMESPACE}
    
    log_info "Waiting for PostgreSQL to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n ${NAMESPACE} --timeout=300s
    
    log_info "PostgreSQL is ready"
}

# Deploy Redis
deploy_redis() {
    log_info "Deploying Redis..."
    kubectl apply -f k8s/deployment.yaml -n ${NAMESPACE}
    
    log_info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis -n ${NAMESPACE} --timeout=300s
    
    log_info "Redis is ready"
}

# Initialize database
init_database() {
    log_info "Initializing database..."
    
    # Get PostgreSQL pod name
    POSTGRES_POD=$(kubectl get pods -n ${NAMESPACE} -l app=postgres -o jsonpath='{.items[0].metadata.name}')
    
    # Copy init script to pod
    kubectl cp init_database.py ${NAMESPACE}/${POSTGRES_POD}:/tmp/init_database.py
    
    # Run init script
    kubectl exec -n ${NAMESPACE} ${POSTGRES_POD} -- python /tmp/init_database.py
    
    log_info "Database initialized"
}

# Deploy web application
deploy_web() {
    log_info "Deploying web application..."
    kubectl apply -f k8s/deployment.yaml -n ${NAMESPACE}
    
    log_info "Waiting for web application to be ready..."
    kubectl wait --for=condition=ready pod -l app=video-processing-web -n ${NAMESPACE} --timeout=300s
    
    log_info "Web application is ready"
}

# Deploy Celery workers
deploy_workers() {
    log_info "Deploying Celery workers..."
    kubectl apply -f k8s/deployment.yaml -n ${NAMESPACE}
    
    log_info "Waiting for workers to be ready..."
    kubectl wait --for=condition=ready pod -l app=video-processing-worker -n ${NAMESPACE} --timeout=300s
    
    log_info "Celery workers are ready"
}

# Deploy Flower monitoring
deploy_flower() {
    log_info "Deploying Flower monitoring..."
    kubectl apply -f k8s/deployment.yaml -n ${NAMESPACE}
    
    log_info "Waiting for Flower to be ready..."
    kubectl wait --for=condition=ready pod -l app=flower -n ${NAMESPACE} --timeout=300s
    
    log_info "Flower monitoring is ready"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    kubectl apply -f k8s/services.yaml -n ${NAMESPACE}
    log_info "Services deployed"
}

# Deploy autoscaling
deploy_autoscaling() {
    log_info "Deploying autoscaling configurations..."
    kubectl apply -f k8s/autoscaling.yaml -n ${NAMESPACE}
    log_info "Autoscaling configurations deployed"
}

# Get service URLs
get_urls() {
    log_info "Getting service URLs..."
    
    # Get web service URL
    WEB_URL=$(kubectl get service video-processing-web-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$WEB_URL" ]; then
        WEB_URL=$(kubectl get service video-processing-web-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    # Get Flower URL
    FLOWER_URL=$(kubectl get service flower-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$FLOWER_URL" ]; then
        FLOWER_URL=$(kubectl get service flower-service -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    log_info "Web application URL: http://${WEB_URL}"
    log_info "Flower monitoring URL: http://${FLOWER_URL}:5555"
}

# Check deployment status
check_status() {
    log_info "Checking deployment status..."
    
    # Check pods
    kubectl get pods -n ${NAMESPACE}
    
    # Check services
    kubectl get services -n ${NAMESPACE}
    
    # Check deployments
    kubectl get deployments -n ${NAMESPACE}
    
    # Check HPA
    kubectl get hpa -n ${NAMESPACE}
}

# Cleanup function
cleanup() {
    log_warn "Cleaning up deployment..."
    kubectl delete namespace ${NAMESPACE}
    log_info "Cleanup completed"
}

# Main deployment function
deploy() {
    log_info "Starting deployment..."
    
    check_kubectl
    check_docker
    build_image
    push_image "$1"
    create_namespace
    deploy_postgres
    deploy_redis
    init_database
    deploy_web
    deploy_workers
    deploy_flower
    deploy_services
    deploy_autoscaling
    
    log_info "Deployment completed successfully!"
    get_urls
    check_status
}

# Main script
case "$1" in
    "deploy")
        deploy "$2"
        ;;
    "status")
        check_status
        ;;
    "cleanup")
        cleanup
        ;;
    "logs")
        kubectl logs -f -l app=video-processing-web -n ${NAMESPACE}
        ;;
    "worker-logs")
        kubectl logs -f -l app=video-processing-worker -n ${NAMESPACE}
        ;;
    "scale-workers")
        kubectl scale deployment video-processing-worker --replicas=$2 -n ${NAMESPACE}
        ;;
    "scale-web")
        kubectl scale deployment video-processing-web --replicas=$2 -n ${NAMESPACE}
        ;;
    *)
        echo "Usage: $0 {deploy|status|cleanup|logs|worker-logs|scale-workers|scale-web}"
        echo ""
        echo "Commands:"
        echo "  deploy [--push]    Deploy the application"
        echo "  status             Check deployment status"
        echo "  cleanup            Clean up the deployment"
        echo "  logs               View web application logs"
        echo "  worker-logs        View worker logs"
        echo "  scale-workers N    Scale workers to N replicas"
        echo "  scale-web N        Scale web app to N replicas"
        exit 1
        ;;
esac
