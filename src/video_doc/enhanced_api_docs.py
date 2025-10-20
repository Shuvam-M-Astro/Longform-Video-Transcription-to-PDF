"""
Interactive API documentation with Swagger UI integration.
"""

import json
from flask import Blueprint, render_template, jsonify, request, Response
from .api_docs import api_docs
from .flask_auth import require_auth, optional_auth, get_current_user_session
from .auth import Permission

def create_api_docs_blueprint() -> Blueprint:
    """Create enhanced Flask blueprint for API documentation."""
    bp = Blueprint('api_docs', __name__, url_prefix='/api')
    
    @bp.route('/docs')
    @optional_auth
    def api_documentation():
        """Interactive API documentation page with Swagger UI."""
        user_session = get_current_user_session()
        return render_template('api_docs.html', 
                             endpoints=api_docs.endpoints,
                             user=user_session)
    
    @bp.route('/openapi.json')
    def openapi_spec():
        """OpenAPI specification endpoint."""
        return jsonify(api_docs.get_openapi_spec())
    
    @bp.route('/endpoints')
    @optional_auth
    def list_endpoints():
        """List all API endpoints with user permissions."""
        user_session = get_current_user_session()
        
        endpoints_data = []
        for endpoint in api_docs.endpoints:
            # Check if user has permission for this endpoint
            accessible = True
            if endpoint.auth_required and user_session:
                accessible = all(
                    auth_manager.has_permission(user_session, Permission(p))
                    for p in endpoint.permissions
                )
            elif endpoint.auth_required and not user_session:
                accessible = False
            
            endpoints_data.append({
                'path': endpoint.path,
                'method': endpoint.method.value,
                'summary': endpoint.summary,
                'tags': endpoint.tags,
                'auth_required': endpoint.auth_required,
                'permissions': endpoint.permissions,
                'accessible': accessible,
                'examples': endpoint.examples
            })
        
        return jsonify({
            'endpoints': endpoints_data,
            'total': len(endpoints_data),
            'accessible': len([ep for ep in endpoints_data if ep['accessible']]),
            'user': {
                'authenticated': user_session is not None,
                'role': user_session.role.value if user_session else None,
                'permissions': [p.value for p in user_session.permissions] if user_session else []
            }
        })
    
    @bp.route('/endpoints/<tag>')
    @optional_auth
    def endpoints_by_tag(tag: str):
        """Get endpoints by tag with permission filtering."""
        user_session = get_current_user_session()
        filtered_endpoints = api_docs.get_endpoints_by_tag(tag)
        
        # Filter by user permissions
        accessible_endpoints = []
        for endpoint in filtered_endpoints:
            accessible = True
            if endpoint.auth_required and user_session:
                accessible = all(
                    auth_manager.has_permission(user_session, Permission(p))
                    for p in endpoint.permissions
                )
            elif endpoint.auth_required and not user_session:
                accessible = False
            
            if accessible:
                accessible_endpoints.append(endpoint)
        
        return jsonify({
            'tag': tag,
            'endpoints': [asdict(ep) for ep in accessible_endpoints],
            'count': len(accessible_endpoints),
            'total': len(filtered_endpoints)
        })
    
    @bp.route('/test-endpoint', methods=['POST'])
    @require_auth()
    def test_endpoint():
        """Test endpoint for API documentation."""
        try:
            data = request.get_json()
            endpoint_path = data.get('path')
            method = data.get('method', 'GET')
            
            # This would integrate with actual API testing
            # For now, return a mock response
            return jsonify({
                'message': 'Endpoint test successful',
                'path': endpoint_path,
                'method': method,
                'timestamp': datetime.now().isoformat(),
                'note': 'This is a mock response for documentation purposes'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @bp.route('/generate-client-code')
    @require_auth(Permission.VIEW_METRICS)
    def generate_client_code():
        """Generate client code examples."""
        language = request.args.get('language', 'python')
        
        if language == 'python':
            code = '''
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
'''
        elif language == 'javascript':
            code = '''
class VideoProcessorClient {
    constructor(baseUrl, apiKey = null, sessionToken = null) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json'
        };
        
        if (apiKey) {
            this.headers['X-API-Key'] = apiKey;
        } else if (sessionToken) {
            this.headers['Authorization'] = `Bearer ${sessionToken}`;
        }
    }
    
    async createJob(url) {
        const response = await fetch(`${this.baseUrl}/process_url`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ url })
        });
        return response.json();
    }
    
    async getJobs() {
        const response = await fetch(`${this.baseUrl}/jobs`, {
            headers: this.headers
        });
        return response.json();
    }
}

// Usage
const client = new VideoProcessorClient('http://localhost:5000', 'your_api_key');
const jobs = await client.getJobs();
'''
        else:
            code = 'Language not supported yet'
        
        return Response(code, mimetype='text/plain')
    
    return bp
