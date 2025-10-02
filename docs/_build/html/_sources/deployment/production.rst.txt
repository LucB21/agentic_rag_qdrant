Production Deployment
====================

This guide covers deploying Agentic RAG to production environments with best practices for scalability, security, and reliability.

Deployment Overview
-------------------

Production deployment involves several components:

- **Application Server**: Hosting the Agentic RAG application
- **Vector Database**: Qdrant cluster for scalable search
- **Load Balancer**: Distribution of requests across instances
- **Monitoring**: System health and performance tracking
- **CI/CD Pipeline**: Automated testing and deployment

Architecture Options
--------------------

Single Server Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~

**Suitable for**: Small teams, development environments, proof of concepts

.. code-block:: text

   ┌─────────────────────────────────────┐
   │          Single Server              │
   │                                     │
   │  ┌─────────────┐  ┌─────────────┐   │
   │  │  Agentic    │  │   Qdrant    │   │
   │  │    RAG      │  │   Local     │   │
   │  │             │  │             │   │
   │  └─────────────┘  └─────────────┘   │
   │                                     │
   │  ┌─────────────┐  ┌─────────────┐   │
   │  │  Streamlit  │  │ Monitoring  │   │
   │  │   Server    │  │   (Opik)    │   │
   │  └─────────────┘  └─────────────┘   │
   └─────────────────────────────────────┘

**Pros**: Simple setup, cost-effective, easy debugging
**Cons**: Single point of failure, limited scalability

Multi-Server Deployment
~~~~~~~~~~~~~~~~~~~~~~~

**Suitable for**: Production workloads, high availability requirements

.. code-block:: text

   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │  Load Balancer  │    │   Monitoring    │    │   CI/CD Server  │
   │    (nginx)      │    │   Dashboard     │    │    (GitHub)     │
   └─────────┬───────┘    └─────────────────┘    └─────────────────┘
             │
   ┌─────────▼─────────────────────────────────────────────────────┐
   │                    Application Layer                          │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
   │  │ App Server  │  │ App Server  │  │ App Server  │          │
   │  │     #1      │  │     #2      │  │     #3      │          │
   │  └─────────────┘  └─────────────┘  └─────────────┘          │
   └───────────┬───────────────────────────────────────────────────┘
               │
   ┌───────────▼───────────────────────────────────────────────────┐
   │                     Data Layer                                │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
   │  │   Qdrant    │  │   Qdrant    │  │   Qdrant    │          │
   │  │  Primary    │  │  Replica    │  │  Replica    │          │
   │  └─────────────┘  └─────────────┘  └─────────────┘          │
   └───────────────────────────────────────────────────────────────┘

**Pros**: High availability, scalable, fault tolerant
**Cons**: Complex setup, higher costs, more maintenance

Cloud Deployment
~~~~~~~~~~~~~~~~

**Suitable for**: Enterprise deployments, global scale, managed services

Options include:
- **Azure Container Instances (ACI)**
- **Azure Kubernetes Service (AKS)**
- **AWS ECS/EKS**
- **Google Cloud Run/GKE**

Docker Deployment
-----------------

Basic Docker Setup
~~~~~~~~~~~~~~~~~~

Create `Dockerfile`:

.. code-block:: dockerfile

   FROM python:3.10-slim

   # Set working directory
   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       g++ \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY src/ ./src/
   COPY streamlit_app.py .
   COPY .env .

   # Expose ports
   EXPOSE 8501 8000

   # Health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
     CMD curl -f http://localhost:8501 || exit 1

   # Start command
   CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]

Multi-Service Docker Compose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create `docker-compose.prod.yml`:

.. code-block:: yaml

   version: '3.8'

   services:
     qdrant:
       image: qdrant/qdrant:latest
       container_name: qdrant
       ports:
         - "6333:6333"
         - "6334:6334"
       volumes:
         - qdrant_storage:/qdrant/storage
       environment:
         - QDRANT__SERVICE__HTTP_PORT=6333
         - QDRANT__SERVICE__GRPC_PORT=6334
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
         interval: 30s
         timeout: 10s
         retries: 3

     agentic-rag:
       build: .
       container_name: agentic-rag
       ports:
         - "8501:8501"
       environment:
         - QDRANT_URL=http://qdrant:6333
         - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
         - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
       depends_on:
         qdrant:
           condition: service_healthy
       restart: unless-stopped
       volumes:
         - ./logs:/app/logs
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8501"]
         interval: 30s
         timeout: 10s
         retries: 3

     nginx:
       image: nginx:alpine
       container_name: nginx
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
         - ./ssl:/etc/nginx/ssl
       depends_on:
         - agentic-rag
       restart: unless-stopped

     monitoring:
       image: opik-monitoring:latest
       container_name: monitoring
       ports:
         - "3000:3000"
       environment:
         - DATABASE_URL=postgresql://user:pass@db:5432/opik
       restart: unless-stopped

   volumes:
     qdrant_storage:
       driver: local

Nginx Configuration
~~~~~~~~~~~~~~~~~~~

Create `nginx.conf`:

.. code-block:: nginx

   events {
       worker_connections 1024;
   }

   http {
       upstream agentic-rag {
           server agentic-rag:8501;
       }

       server {
           listen 80;
           server_name your-domain.com;
           
           # Redirect HTTP to HTTPS
           return 301 https://$host$request_uri;
       }

       server {
           listen 443 ssl http2;
           server_name your-domain.com;

           ssl_certificate /etc/nginx/ssl/cert.pem;
           ssl_certificate_key /etc/nginx/ssl/key.pem;

           # Security headers
           add_header X-Frame-Options DENY;
           add_header X-Content-Type-Options nosniff;
           add_header X-XSS-Protection "1; mode=block";

           # Proxy settings
           location / {
               proxy_pass http://agentic-rag;
               proxy_set_header Host $host;
               proxy_set_header X-Real-IP $remote_addr;
               proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
               proxy_set_header X-Forwarded-Proto $scheme;
               
               # WebSocket support for Streamlit
               proxy_http_version 1.1;
               proxy_set_header Upgrade $http_upgrade;
               proxy_set_header Connection "upgrade";
           }

           # Health check endpoint
           location /health {
               access_log off;
               return 200 "healthy\n";
               add_header Content-Type text/plain;
           }
       }
   }

Kubernetes Deployment
----------------------

Kubernetes Manifests
~~~~~~~~~~~~~~~~~~~~

Create `k8s/namespace.yaml`:

.. code-block:: yaml

   apiVersion: v1
   kind: Namespace
   metadata:
     name: agentic-rag

Create `k8s/configmap.yaml`:

.. code-block:: yaml

   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: agentic-rag-config
     namespace: agentic-rag
   data:
     QDRANT_URL: "http://qdrant-service:6333"
     AZURE_OPENAI_API_VERSION: "2024-02-15-preview"
     LOG_LEVEL: "INFO"

Create `k8s/secret.yaml`:

.. code-block:: yaml

   apiVersion: v1
   kind: Secret
   metadata:
     name: agentic-rag-secrets
     namespace: agentic-rag
   type: Opaque
   data:
     AZURE_OPENAI_API_KEY: <base64-encoded-key>
     AZURE_OPENAI_ENDPOINT: <base64-encoded-endpoint>
     QDRANT_API_KEY: <base64-encoded-qdrant-key>

Create `k8s/qdrant-deployment.yaml`:

.. code-block:: yaml

   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: qdrant
     namespace: agentic-rag
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: qdrant
     template:
       metadata:
         labels:
           app: qdrant
       spec:
         containers:
         - name: qdrant
           image: qdrant/qdrant:latest
           ports:
           - containerPort: 6333
           - containerPort: 6334
           env:
           - name: QDRANT__SERVICE__HTTP_PORT
             value: "6333"
           - name: QDRANT__SERVICE__GRPC_PORT
             value: "6334"
           volumeMounts:
           - name: qdrant-storage
             mountPath: /qdrant/storage
           livenessProbe:
             httpGet:
               path: /health
               port: 6333
             initialDelaySeconds: 30
             periodSeconds: 30
           readinessProbe:
             httpGet:
               path: /readiness
               port: 6333
             initialDelaySeconds: 5
             periodSeconds: 5
         volumes:
         - name: qdrant-storage
           persistentVolumeClaim:
             claimName: qdrant-pvc

Create `k8s/app-deployment.yaml`:

.. code-block:: yaml

   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: agentic-rag
     namespace: agentic-rag
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: agentic-rag
     template:
       metadata:
         labels:
           app: agentic-rag
       spec:
         containers:
         - name: agentic-rag
           image: your-registry/agentic-rag:latest
           ports:
           - containerPort: 8501
           env:
           - name: QDRANT_URL
             valueFrom:
               configMapKeyRef:
                 name: agentic-rag-config
                 key: QDRANT_URL
           - name: AZURE_OPENAI_API_KEY
             valueFrom:
               secretKeyRef:
                 name: agentic-rag-secrets
                 key: AZURE_OPENAI_API_KEY
           - name: AZURE_OPENAI_ENDPOINT
             valueFrom:
               secretKeyRef:
                 name: agentic-rag-secrets
                 key: AZURE_OPENAI_ENDPOINT
           resources:
             requests:
               memory: "1Gi"
               cpu: "500m"
             limits:
               memory: "2Gi"
               cpu: "1000m"
           livenessProbe:
             httpGet:
               path: /
               port: 8501
             initialDelaySeconds: 60
             periodSeconds: 30
           readinessProbe:
             httpGet:
               path: /
               port: 8501
             initialDelaySeconds: 10
             periodSeconds: 5

Helm Chart Deployment
~~~~~~~~~~~~~~~~~~~~~

Create `helm/agentic-rag/values.yaml`:

.. code-block:: yaml

   replicaCount: 3

   image:
     repository: your-registry/agentic-rag
     tag: latest
     pullPolicy: Always

   service:
     type: ClusterIP
     port: 8501

   ingress:
     enabled: true
     className: nginx
     annotations:
       cert-manager.io/cluster-issuer: letsencrypt-prod
     hosts:
     - host: agentic-rag.your-domain.com
       paths:
       - path: /
         pathType: Prefix
     tls:
     - secretName: agentic-rag-tls
       hosts:
       - agentic-rag.your-domain.com

   qdrant:
     enabled: true
     replicas: 3
     storage:
       size: 100Gi
       storageClass: fast-ssd

   monitoring:
     enabled: true
     opik:
       enabled: true

   autoscaling:
     enabled: true
     minReplicas: 2
     maxReplicas: 10
     targetCPUUtilizationPercentage: 70

Cloud-Specific Deployments
---------------------------

Azure Container Instances
~~~~~~~~~~~~~~~~~~~~~~~~~

Create `azure-deployment.yml`:

.. code-block:: yaml

   apiVersion: 2019-12-01
   location: eastus
   name: agentic-rag-container-group
   properties:
     containers:
     - name: agentic-rag
       properties:
         image: your-registry/agentic-rag:latest
         resources:
           requests:
             cpu: 2
             memoryInGb: 4
         ports:
         - port: 8501
           protocol: TCP
         environmentVariables:
         - name: AZURE_OPENAI_API_KEY
           secureValue: your-api-key
         - name: AZURE_OPENAI_ENDPOINT
           value: your-endpoint
         - name: QDRANT_URL
           value: https://your-qdrant-cluster.qdrant.app
     - name: qdrant
       properties:
         image: qdrant/qdrant:latest
         resources:
           requests:
             cpu: 1
             memoryInGb: 2
         ports:
         - port: 6333
           protocol: TCP
     osType: Linux
     ipAddress:
       type: Public
       ports:
       - protocol: TCP
         port: 8501
       dnsNameLabel: agentic-rag-demo

Deploy with Azure CLI:

.. code-block:: bash

   az container create --resource-group myResourceGroup \
     --file azure-deployment.yml

AWS ECS Deployment
~~~~~~~~~~~~~~~~~~

Create `ecs-task-definition.json`:

.. code-block:: json

   {
     "family": "agentic-rag",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "2048",
     "memory": "4096",
     "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "agentic-rag",
         "image": "your-registry/agentic-rag:latest",
         "portMappings": [
           {
             "containerPort": 8501,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "AZURE_OPENAI_ENDPOINT",
             "value": "your-endpoint"
           }
         ],
         "secrets": [
           {
             "name": "AZURE_OPENAI_API_KEY",
             "valueFrom": "arn:aws:ssm:region:account:parameter/agentic-rag/api-key"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/agentic-rag",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }

Security Best Practices
------------------------

Environment Security
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Use secrets management
   export AZURE_OPENAI_API_KEY="$(kubectl get secret agentic-rag-secrets -o jsonpath='{.data.AZURE_OPENAI_API_KEY}' | base64 -d)"

   # Network security
   # - Use private networks/VPCs
   # - Configure firewalls and security groups
   # - Enable HTTPS/TLS encryption
   # - Use API gateways for rate limiting

Application Security
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Input validation
   from pydantic import BaseModel, validator

   class QueryRequest(BaseModel):
       question: str
       sector: List[str]
       
       @validator('question')
       def validate_question(cls, v):
           if len(v) > 500:
               raise ValueError('Question too long')
           return v.strip()

   # Rate limiting
   from slowapi import Limiter
   from slowapi.util import get_remote_address

   limiter = Limiter(key_func=get_remote_address)

   @app.post("/query")
   @limiter.limit("10/minute")
   async def query_endpoint(request: Request, query: QueryRequest):
       # Process query
       pass

Monitoring and Observability
-----------------------------

Application Monitoring
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prometheus metrics
   from prometheus_client import Counter, Histogram, generate_latest

   REQUEST_COUNT = Counter('agentic_rag_requests_total', 'Total requests')
   REQUEST_DURATION = Histogram('agentic_rag_request_duration_seconds', 'Request duration')

   @REQUEST_DURATION.time()
   def process_query(question, sector):
       REQUEST_COUNT.inc()
       # Process query
       pass

   # Health check endpoint
   @app.get("/health")
   async def health_check():
       try:
           # Check Qdrant connection
           qdrant_healthy = check_qdrant_health()
           # Check Azure OpenAI
           openai_healthy = check_openai_health()
           
           if qdrant_healthy and openai_healthy:
               return {"status": "healthy"}
           else:
               return {"status": "unhealthy"}, 503
       except Exception as e:
           return {"status": "error", "message": str(e)}, 503

Logging Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging
   import json
   from datetime import datetime

   class StructuredLogger:
       def __init__(self):
           self.logger = logging.getLogger("agentic_rag")
           handler = logging.StreamHandler()
           handler.setFormatter(StructuredFormatter())
           self.logger.addHandler(handler)
           self.logger.setLevel(logging.INFO)

   class StructuredFormatter(logging.Formatter):
       def format(self, record):
           log_entry = {
               "timestamp": datetime.utcnow().isoformat(),
               "level": record.levelname,
               "message": record.getMessage(),
               "module": record.module,
               "function": record.funcName,
               "line": record.lineno
           }
           return json.dumps(log_entry)

CI/CD Pipeline
--------------

GitHub Actions Workflow
~~~~~~~~~~~~~~~~~~~~~~~

Create `.github/workflows/deploy.yml`:

.. code-block:: yaml

   name: Deploy to Production

   on:
     push:
       branches: [main]
     workflow_dispatch:

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python
         uses: actions/setup-python@v3
         with:
           python-version: '3.10'
       
       - name: Install dependencies
         run: |
           pip install -r requirements.txt
       
       - name: Run tests
         run: |
           python -m pytest tests/
       
       - name: Run RAGAS evaluation
         env:
           AZURE_OPENAI_API_KEY: ${{ secrets.AZURE_OPENAI_API_KEY }}
         run: |
           cd test_ragas
           python rag_ragas_qdrant.py

     build:
       needs: test
       runs-on: ubuntu-latest
       steps:
       - uses: actions/checkout@v3
       
       - name: Build and push Docker image
         uses: docker/build-push-action@v2
         with:
           context: .
           push: true
           tags: ${{ secrets.REGISTRY }}/agentic-rag:${{ github.sha }}

     deploy:
       needs: build
       runs-on: ubuntu-latest
       steps:
       - name: Deploy to Kubernetes
         uses: azure/k8s-deploy@v1
         with:
           manifests: |
             k8s/deployment.yaml
             k8s/service.yaml
           images: |
             ${{ secrets.REGISTRY }}/agentic-rag:${{ github.sha }}

Performance Optimization
-------------------------

Application Performance
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Connection pooling
   from qdrant_client import QdrantClient
   from functools import lru_cache

   @lru_cache(maxsize=1)
   def get_qdrant_client():
       return QdrantClient(
           url=os.getenv("QDRANT_URL"),
           api_key=os.getenv("QDRANT_API_KEY"),
           prefer_grpc=True,
           timeout=30
       )

   # Caching
   from functools import lru_cache
   import hashlib

   @lru_cache(maxsize=1000)
   def cached_search(query_hash, collection_name, top_k):
       # Perform search
       pass

   def search_with_cache(query, collection_name, top_k=10):
       query_hash = hashlib.md5(query.encode()).hexdigest()
       return cached_search(query_hash, collection_name, top_k)

Database Optimization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Qdrant optimization settings
   QDRANT_CONFIG = {
       "search_params": {
           "hnsw_ef": 128,  # Higher for better accuracy
           "exact": False   # Use approximate search
       },
       "quantization": {
           "type": "scalar",
           "quantile": 0.99
       },
       "optimizer_config": {
           "deleted_threshold": 0.2,
           "vacuum_min_vector_number": 1000,
           "default_segment_number": 2
       }
   }

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**1. Out of Memory Errors**

.. code-block:: yaml

   # Increase memory limits in Kubernetes
   resources:
     limits:
       memory: "4Gi"
     requests:
       memory: "2Gi"

**2. Slow Response Times**

.. code-block:: python

   # Enable connection pooling and caching
   # Use async/await for better concurrency
   # Optimize Qdrant search parameters

**3. API Rate Limits**

.. code-block:: python

   # Implement exponential backoff
   import time
   import random

   def retry_with_backoff(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return func()
           except RateLimitError:
               if attempt == max_retries - 1:
                   raise
               time.sleep(2 ** attempt + random.uniform(0, 1))

Monitoring Checklist
~~~~~~~~~~~~~~~~~~~

- [ ] Application health checks working
- [ ] Database connections monitored
- [ ] API response times tracked
- [ ] Error rates below threshold
- [ ] Resource usage within limits
- [ ] Logs properly structured and collected
- [ ] Alerts configured for critical issues

Next Steps
----------

After deployment:

1. **Monitor Performance**: Set up comprehensive monitoring
2. **Optimize Costs**: Review resource usage and optimize
3. **Scale Testing**: Test system under load
4. **Security Audit**: Regular security reviews
5. **Backup Strategy**: Implement data backup and recovery