Docker Deployment
=================

This guide covers containerizing and deploying Agentic RAG using Docker and Docker Compose.

Overview
--------

Docker deployment provides:

- **Consistency**: Same environment across development, testing, and production
- **Isolation**: Applications run in isolated containers
- **Scalability**: Easy horizontal scaling with container orchestration
- **Portability**: Deploy anywhere Docker is supported

Prerequisites
-------------

- Docker Engine 20.0+
- Docker Compose 2.0+
- 8GB+ RAM recommended
- SSL certificates (for production)

Basic Docker Setup
------------------

Single Container Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create `Dockerfile`:

.. code-block:: dockerfile

   # Use Python 3.10 slim image
   FROM python:3.10-slim

   # Set environment variables
   ENV PYTHONDONTWRITEBYTECODE=1
   ENV PYTHONUNBUFFERED=1
   ENV PYTHONPATH=/app/src

   # Set working directory
   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       g++ \
       curl \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements first for better caching
   COPY requirements.txt .

   # Install Python dependencies
   RUN pip install --no-cache-dir --upgrade pip && \
       pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY src/ ./src/
   COPY streamlit_app.py .
   COPY test_ragas/ ./test_ragas/

   # Create non-root user
   RUN groupadd -r appuser && useradd -r -g appuser appuser
   RUN chown -R appuser:appuser /app
   USER appuser

   # Expose ports
   EXPOSE 8501

   # Health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
     CMD curl -f http://localhost:8501/_stcore/health || exit 1

   # Default command
   CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

Build and Run
~~~~~~~~~~~~~

.. code-block:: bash

   # Build the image
   docker build -t agentic-rag:latest .

   # Run with environment variables
   docker run -d \
     --name agentic-rag \
     -p 8501:8501 \
     -e AZURE_OPENAI_API_KEY="your_api_key" \
     -e AZURE_OPENAI_ENDPOINT="your_endpoint" \
     -e QDRANT_URL="http://host.docker.internal:6333" \
     agentic-rag:latest

Multi-Container Setup
---------------------

Docker Compose Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create `docker-compose.yml`:

.. code-block:: yaml

   version: '3.8'

   services:
     qdrant:
       image: qdrant/qdrant:v1.7.4
       container_name: qdrant
       ports:
         - "6333:6333"
         - "6334:6334"
       volumes:
         - qdrant_storage:/qdrant/storage
         - ./qdrant_config.yaml:/qdrant/config/production.yaml
       environment:
         - QDRANT__SERVICE__HTTP_PORT=6333
         - QDRANT__SERVICE__GRPC_PORT=6334
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
         interval: 30s
         timeout: 10s
         retries: 3
         start_period: 40s

     agentic-rag:
       build: .
       container_name: agentic-rag
       ports:
         - "8501:8501"
       environment:
         - QDRANT_URL=http://qdrant:6333
         - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
         - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
         - AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-2024-02-15-preview}
         - AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=${AZURE_OPENAI_CHAT_DEPLOYMENT_NAME:-gpt-4o}
         - AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=${AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME:-text-embedding-ada-002}
       depends_on:
         qdrant:
           condition: service_healthy
       restart: unless-stopped
       volumes:
         - ./logs:/app/logs
         - ./data:/app/data
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
         interval: 30s
         timeout: 10s
         retries: 3
         start_period: 60s

   volumes:
     qdrant_storage:
       driver: local

Environment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

Create `.env` file:

.. code-block:: bash

   # Azure OpenAI Configuration
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o
   AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=text-embedding-ada-002

   # Optional: OpenAI for web search
   OPENAI_API_KEY=your_openai_key

   # Logging
   LOG_LEVEL=INFO

   # Performance
   QDRANT_TIMEOUT=30
   MAX_CONCURRENT_REQUESTS=10

Qdrant Configuration
~~~~~~~~~~~~~~~~~~~

Create `qdrant_config.yaml`:

.. code-block:: yaml

   service:
     http_port: 6333
     grpc_port: 6334
     enable_cors: true

   storage:
     # Storage configuration
     storage_path: ./storage
     
     # Performance optimizations
     optimizers:
       default_segment_number: 2
       memmap_threshold: 200000
       indexing_threshold: 10000

   # Cluster configuration (for multi-node setup)
   cluster:
     enabled: false

   # Telemetry
   telemetry:
     disabled: true

Production Deployment
--------------------

Production Docker Compose
~~~~~~~~~~~~~~~~~~~~~~~~~

Create `docker-compose.prod.yml`:

.. code-block:: yaml

   version: '3.8'

   services:
     qdrant:
       image: qdrant/qdrant:v1.7.4
       container_name: qdrant-prod
       ports:
         - "6333:6333"
       volumes:
         - qdrant_storage_prod:/qdrant/storage
         - ./qdrant_prod_config.yaml:/qdrant/config/production.yaml
       environment:
         - QDRANT__SERVICE__HTTP_PORT=6333
         - QDRANT__LOG_LEVEL=INFO
       restart: unless-stopped
       deploy:
         resources:
           limits:
             memory: 4G
             cpus: '2.0'
           reservations:
             memory: 2G
             cpus: '1.0'
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
         interval: 30s
         timeout: 10s
         retries: 3

     agentic-rag:
       build:
         context: .
         dockerfile: Dockerfile.prod
       container_name: agentic-rag-prod
       environment:
         - QDRANT_URL=http://qdrant:6333
         - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
         - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
         - LOG_LEVEL=WARNING
         - ENVIRONMENT=production
       depends_on:
         qdrant:
           condition: service_healthy
       restart: unless-stopped
       deploy:
         replicas: 3
         resources:
           limits:
             memory: 2G
             cpus: '1.0'
           reservations:
             memory: 1G
             cpus: '0.5'
       volumes:
         - app_logs:/app/logs
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
         interval: 30s
         timeout: 10s
         retries: 3

     nginx:
       image: nginx:alpine
       container_name: nginx-proxy
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
         - ./nginx/ssl:/etc/nginx/ssl:ro
         - nginx_logs:/var/log/nginx
       depends_on:
         - agentic-rag
       restart: unless-stopped

     prometheus:
       image: prom/prometheus:latest
       container_name: prometheus
       ports:
         - "9090:9090"
       volumes:
         - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
         - prometheus_data:/prometheus
       command:
         - '--config.file=/etc/prometheus/prometheus.yml'
         - '--storage.tsdb.path=/prometheus'
         - '--web.console.libraries=/etc/prometheus/console_libraries'
         - '--web.console.templates=/etc/prometheus/consoles'
       restart: unless-stopped

     grafana:
       image: grafana/grafana:latest
       container_name: grafana
       ports:
         - "3000:3000"
       environment:
         - GF_SECURITY_ADMIN_PASSWORD=admin
       volumes:
         - grafana_data:/var/lib/grafana
         - ./monitoring/grafana:/etc/grafana/provisioning
       restart: unless-stopped

   volumes:
     qdrant_storage_prod:
     app_logs:
     nginx_logs:
     prometheus_data:
     grafana_data:

Production Dockerfile
~~~~~~~~~~~~~~~~~~~~~

Create `Dockerfile.prod`:

.. code-block:: dockerfile

   FROM python:3.10-slim

   # Production environment variables
   ENV PYTHONDONTWRITEBYTECODE=1
   ENV PYTHONUNBUFFERED=1
   ENV PYTHONPATH=/app/src
   ENV ENVIRONMENT=production

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       g++ \
       curl \
       && rm -rf /var/lib/apt/lists/* \
       && apt-get clean

   # Copy and install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir --upgrade pip && \
       pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY src/ ./src/
   COPY streamlit_app.py .

   # Create non-root user
   RUN groupadd -r appuser && useradd -r -g appuser appuser
   RUN mkdir -p /app/logs && chown -R appuser:appuser /app
   USER appuser

   # Expose port
   EXPOSE 8501

   # Health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
     CMD curl -f http://localhost:8501/_stcore/health || exit 1

   # Production command with optimized settings
   CMD ["streamlit", "run", "streamlit_app.py", \
        "--server.address", "0.0.0.0", \
        "--server.port", "8501", \
        "--server.enableCORS", "false", \
        "--server.enableXsrfProtection", "true", \
        "--browser.gatherUsageStats", "false"]

Load Balancer Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create `nginx/nginx.conf`:

.. code-block:: nginx

   events {
       worker_connections 1024;
   }

   http {
       upstream agentic_rag_backend {
           least_conn;
           server agentic-rag-prod:8501 max_fails=3 fail_timeout=30s;
           # Add more backend servers for load balancing
           # server agentic-rag-prod-2:8501 max_fails=3 fail_timeout=30s;
           # server agentic-rag-prod-3:8501 max_fails=3 fail_timeout=30s;
       }

       # Rate limiting
       limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;

       server {
           listen 80;
           server_name your-domain.com;
           return 301 https://$server_name$request_uri;
       }

       server {
           listen 443 ssl http2;
           server_name your-domain.com;

           ssl_certificate /etc/nginx/ssl/cert.pem;
           ssl_certificate_key /etc/nginx/ssl/key.pem;
           ssl_session_timeout 1d;
           ssl_session_cache shared:SSL:50m;
           ssl_session_tickets off;

           # Security headers
           add_header X-Frame-Options DENY;
           add_header X-Content-Type-Options nosniff;
           add_header X-XSS-Protection "1; mode=block";
           add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

           # API rate limiting
           location /api/ {
               limit_req zone=api burst=20 nodelay;
               proxy_pass http://agentic_rag_backend;
               include proxy_params;
           }

           location / {
               proxy_pass http://agentic_rag_backend;
               proxy_set_header Host $host;
               proxy_set_header X-Real-IP $remote_addr;
               proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
               proxy_set_header X-Forwarded-Proto $scheme;
               
               # WebSocket support
               proxy_http_version 1.1;
               proxy_set_header Upgrade $http_upgrade;
               proxy_set_header Connection "upgrade";
               
               # Timeouts
               proxy_connect_timeout 60s;
               proxy_send_timeout 60s;
               proxy_read_timeout 60s;
           }

           location /health {
               access_log off;
               return 200 "healthy\n";
               add_header Content-Type text/plain;
           }
       }
   }

Monitoring Setup
---------------

Prometheus Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Create `monitoring/prometheus.yml`:

.. code-block:: yaml

   global:
     scrape_interval: 15s
     evaluation_interval: 15s

   rule_files:
     - "rules/*.yml"

   scrape_configs:
     - job_name: 'agentic-rag'
       static_configs:
         - targets: ['agentic-rag:8501']
       metrics_path: '/metrics'
       scrape_interval: 30s

     - job_name: 'qdrant'
       static_configs:
         - targets: ['qdrant:6333']
       metrics_path: '/metrics'

     - job_name: 'nginx'
       static_configs:
         - targets: ['nginx:80']

Grafana Dashboard
~~~~~~~~~~~~~~~~

Create `monitoring/grafana/dashboards/agentic-rag.json`:

.. code-block:: json

   {
     "dashboard": {
       "title": "Agentic RAG Monitoring",
       "panels": [
         {
           "title": "Request Rate",
           "type": "graph",
           "targets": [
             {
               "expr": "rate(agentic_rag_requests_total[5m])",
               "legendFormat": "Requests/sec"
             }
           ]
         },
         {
           "title": "Response Time",
           "type": "graph", 
           "targets": [
             {
               "expr": "histogram_quantile(0.95, rate(agentic_rag_request_duration_seconds_bucket[5m]))",
               "legendFormat": "95th percentile"
             }
           ]
         },
         {
           "title": "Error Rate",
           "type": "graph",
           "targets": [
             {
               "expr": "rate(agentic_rag_errors_total[5m])",
               "legendFormat": "Errors/sec"
             }
           ]
         }
       ]
     }
   }

Deployment Commands
------------------

Development Deployment
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Start development environment
   docker-compose up -d

   # View logs
   docker-compose logs -f agentic-rag

   # Scale services
   docker-compose up -d --scale agentic-rag=3

   # Stop services
   docker-compose down

Production Deployment
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Deploy to production
   docker-compose -f docker-compose.prod.yml up -d

   # Update application
   docker-compose -f docker-compose.prod.yml pull
   docker-compose -f docker-compose.prod.yml up -d --no-deps agentic-rag

   # View production logs
   docker-compose -f docker-compose.prod.yml logs -f

   # Backup data
   docker run --rm \
     -v agentic_rag_qdrant_storage_prod:/source:ro \
     -v $(pwd)/backups:/backup \
     alpine tar czf /backup/qdrant-backup-$(date +%Y%m%d).tar.gz -C /source .

Data Management
---------------

Volume Management
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # List volumes
   docker volume ls

   # Inspect volume
   docker volume inspect agentic_rag_qdrant_storage

   # Backup volume
   docker run --rm \
     -v agentic_rag_qdrant_storage:/data:ro \
     -v $(pwd):/backup \
     alpine tar czf /backup/qdrant-backup.tar.gz -C /data .

   # Restore volume
   docker run --rm \
     -v agentic_rag_qdrant_storage:/data \
     -v $(pwd):/backup \
     alpine tar xzf /backup/qdrant-backup.tar.gz -C /data

Document Indexing
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Index documents in running container
   docker exec -it agentic-rag python -c "
   from agentic_rag.tools.rag_module import index_documents_azure
   index_documents_azure()
   "

   # Run evaluation
   docker exec -it agentic-rag python test_ragas/rag_ragas_qdrant.py

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~

**1. Container Won't Start**

.. code-block:: bash

   # Check logs
   docker logs agentic-rag

   # Check configuration
   docker exec -it agentic-rag env

   # Validate health check
   docker exec -it agentic-rag curl http://localhost:8501/_stcore/health

**2. Qdrant Connection Issues**

.. code-block:: bash

   # Test Qdrant connectivity
   docker exec -it agentic-rag curl http://qdrant:6333/health

   # Check network connectivity
   docker network ls
   docker network inspect agentic_rag_default

**3. Memory Issues**

.. code-block:: bash

   # Monitor resource usage
   docker stats

   # Increase memory limits
   docker-compose -f docker-compose.prod.yml up -d \
     --scale agentic-rag=2 \
     --memory="4g" \
     --cpus="2.0"

**4. SSL Certificate Issues**

.. code-block:: bash

   # Generate self-signed certificates for testing
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout nginx/ssl/key.pem \
     -out nginx/ssl/cert.pem

Debug Commands
~~~~~~~~~~~~~

.. code-block:: bash

   # Interactive shell in container
   docker exec -it agentic-rag /bin/bash

   # Check Python environment
   docker exec -it agentic-rag pip list

   # Test imports
   docker exec -it agentic-rag python -c "import agentic_rag; print('OK')"

   # Check file permissions
   docker exec -it agentic-rag ls -la /app

Scaling and Performance
----------------------

Horizontal Scaling
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Scale application containers
   docker-compose up -d --scale agentic-rag=5

   # Use Docker Swarm for multi-node scaling
   docker swarm init
   docker stack deploy -c docker-compose.prod.yml agentic-rag-stack

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Optimize Docker Compose for performance
   services:
     agentic-rag:
       deploy:
         resources:
           limits:
             memory: 2G
             cpus: '1.0'
         restart_policy:
           condition: on-failure
           max_attempts: 3
       environment:
         - PYTHONOPTIMIZE=1  # Enable Python optimizations
         - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200  # Limit upload size

Health Checks and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: dockerfile

   # Enhanced health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
     CMD curl -f http://localhost:8501/_stcore/health && \
         python -c "from agentic_rag.tools.rag_module import check_qdrant_connection; check_qdrant_connection()" || exit 1

Security Best Practices
-----------------------

Container Security
~~~~~~~~~~~~~~~~~

.. code-block:: dockerfile

   # Use non-root user
   RUN groupadd -r appuser && useradd -r -g appuser appuser
   USER appuser

   # Remove unnecessary packages
   RUN apt-get remove -y gcc g++ && \
       apt-get autoremove -y && \
       rm -rf /var/lib/apt/lists/*

   # Set read-only filesystem
   # Add to docker-compose.yml:
   # read_only: true
   # tmpfs:
   #   - /tmp

Network Security
~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Use custom network
   networks:
     agentic-rag-network:
       driver: bridge
       ipam:
         config:
           - subnet: 172.20.0.0/16

   services:
     qdrant:
       networks:
         - agentic-rag-network
       # Don't expose ports externally in production

Next Steps
----------

After Docker deployment:

1. **Monitor Performance**: Set up comprehensive monitoring with Prometheus/Grafana
2. **Implement Backups**: Regular data backups and disaster recovery
3. **Security Hardening**: Regular security audits and updates
4. **Scale Testing**: Load testing to determine scaling requirements
5. **CI/CD Integration**: Automated testing and deployment pipelines