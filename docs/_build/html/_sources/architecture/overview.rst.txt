Architecture Overview
====================

Agentic RAG is built on a sophisticated multi-agent architecture that combines the power of CrewAI, Qdrant vector database, and Azure OpenAI to deliver intelligent retrieval-augmented generation capabilities.

System Architecture
--------------------

.. code-block:: text

   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │   Streamlit     │    │   CLI Interface │    │  Python API     │
   │   Web App       │    │                 │    │                 │
   └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    │
   ┌────────────────────────────────▼────────────────────────────────┐
   │                        RagFlow                                  │
   │                    (Main Flow Controller)                       │
   └─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────┘
             │         │         │         │         │         │
    ┌────────▼──┐ ┌────▼──┐ ┌────▼──┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐
    │Check Crew │ │  RAG  │ │WebSearch│ │Synth  │ │ Opik  │ │Vector │
    │           │ │ Tool  │ │  Crew   │ │ Crew  │ │Monitor│ │  DB   │
    └───────────┘ └───────┘ └─────────┘ └───────┘ └───────┘ └───────┘
                      │                                         │
             ┌────────▼──────────┐                    ┌────────▼──────────┐
             │   Azure OpenAI    │                    │     Qdrant        │
             │  - GPT-4o Chat    │                    │  - Vector Store   │
             │  - Embeddings     │                    │  - Hybrid Search  │
             └───────────────────┘                    │  - Collections    │
                                                      └───────────────────┘

Core Components
---------------

**1. RagFlow (Main Controller)**
   The central orchestrator that manages the entire RAG process:
   
   - Routes queries to appropriate crews
   - Manages state between different processing stages
   - Handles flow control and decision making
   - Integrates monitoring and logging

**2. Multi-Agent Crews**
   Specialized teams of AI agents for different tasks:
   
   - **Check Crew**: Query relevance analysis and routing
   - **Web Search Crew**: External information retrieval
   - **Synthesis Crew**: Information combination and response generation

**3. RAG Tool**
   Core retrieval component that:
   
   - Performs hybrid search in Qdrant
   - Manages document indexing and chunking
   - Implements Maximum Marginal Relevance (MMR)
   - Handles multi-sector document collections

**4. Vector Database (Qdrant)**
   High-performance vector storage and search:
   
   - HNSW indexing for fast similarity search
   - Hybrid search capabilities (semantic + keyword)
   - Scalar quantization for memory efficiency
   - Multi-collection support for different domains

**5. Language Models (Azure OpenAI)**
   Provides AI capabilities through:
   
   - Chat completion models (GPT-4o)
   - Text embedding models (text-embedding-ada-002)
   - Configurable parameters for different use cases

Data Flow Architecture
----------------------

**Query Processing Pipeline:**

.. mermaid::

   graph TD
       A[User Query] --> B[RagFlow Controller]
       B --> C{Skip Check?}
       C -->|No| D[Check Crew]
       C -->|Yes| E[RAG Tool]
       D --> F{Relevant?}
       F -->|Yes| E[RAG Tool]
       F -->|Partial| G[Web Search Crew]
       F -->|No| G
       E --> H[Qdrant Vector DB]
       H --> I[Retrieved Contexts]
       G --> J[Web Results]
       I --> K[Synthesis Crew]
       J --> K
       K --> L[Final Response]

**Document Indexing Pipeline:**

.. mermaid::

   graph TD
       A[Raw Documents] --> B[Document Loader]
       B --> C[Text Chunking]
       C --> D[Azure OpenAI Embeddings]
       D --> E[Vector Generation]
       E --> F[Qdrant Indexing]
       F --> G[Searchable Collections]

Component Interactions
----------------------

Agent Communication
~~~~~~~~~~~~~~~~~~~

Agents within crews communicate through:

- **Structured Messages**: Predefined schemas for data exchange
- **Context Sharing**: Shared state management across agents
- **Task Dependencies**: Sequential and parallel task execution
- **Error Handling**: Graceful degradation and retry mechanisms

Tool Integration
~~~~~~~~~~~~~~~~

Tools are integrated through:

- **CrewAI Tool Framework**: Standardized tool interface
- **Async Operations**: Non-blocking operations for performance
- **Resource Pooling**: Efficient connection management
- **Caching**: Result caching for frequently accessed data

State Management
~~~~~~~~~~~~~~~~

The system maintains state through:

- **Flow State**: Current processing stage and context
- **Agent Memory**: Individual agent conversation history
- **Shared Context**: Information accessible to all agents
- **Persistent Storage**: Long-term data retention in Qdrant

Scalability Design
------------------

Horizontal Scaling
~~~~~~~~~~~~~~~~~~

The architecture supports horizontal scaling through:

- **Stateless Agents**: Agents can be replicated across instances
- **Load Balancing**: Request distribution across multiple instances
- **Database Sharding**: Qdrant collections can be distributed
- **Async Processing**: Non-blocking operations for better throughput

Vertical Scaling
~~~~~~~~~~~~~~~~

Performance can be improved through:

- **Memory Optimization**: Efficient vector storage and retrieval
- **Connection Pooling**: Database connection management
- **Caching Strategies**: Multi-level caching for frequent queries
- **Model Optimization**: Appropriate model selection for tasks

Performance Characteristics
---------------------------

**Query Response Times:**

- Simple RAG queries: 2-5 seconds
- Complex multi-source queries: 5-15 seconds
- Web-enhanced queries: 10-30 seconds

**Throughput Capabilities:**

- Concurrent users: 50-100 (depending on hardware)
- Queries per minute: 200-500
- Document indexing: 1000 docs/hour

**Memory Requirements:**

- Base system: 4-8 GB RAM
- Vector storage: 1-2 GB per 100k documents
- Model caching: 2-4 GB

Security Architecture
---------------------

**API Security:**
- Token-based authentication for Azure OpenAI
- API key management for Qdrant
- Rate limiting and request validation

**Data Security:**
- Encrypted communication (HTTPS/TLS)
- Secure credential storage (environment variables)
- Optional data anonymization

**Access Control:**
- Role-based access to different document collections
- Query logging and audit trails
- Configurable data retention policies

Monitoring and Observability
-----------------------------

**Performance Monitoring:**
- Opik integration for AI operation tracking
- Response time and throughput metrics
- Error rate and success rate monitoring

**System Health:**
- Database connection health checks
- API endpoint availability monitoring
- Resource utilization tracking

**Business Metrics:**
- Query success rates
- User satisfaction indicators
- Knowledge base coverage analysis

Deployment Architecture
-----------------------

**Development Environment:**

.. code-block:: text

   Developer Machine
   ├── Local Qdrant (Docker)
   ├── Azure OpenAI (Cloud)
   ├── Local Python Environment
   └── Streamlit Dev Server

**Production Environment:**

.. code-block:: text

   Production Infrastructure
   ├── Load Balancer
   ├── Application Servers (Multiple)
   │   ├── Agentic RAG Application
   │   └── Monitoring Agents
   ├── Qdrant Cluster
   │   ├── Primary Node
   │   └── Replica Nodes
   └── External Services
       ├── Azure OpenAI
       └── Monitoring Dashboard

Extensibility Points
--------------------

The architecture is designed for extensibility:

**New Crews:**
- Additional specialized agent teams
- Domain-specific processing workflows
- Custom task orchestration patterns

**New Tools:**
- Additional retrieval methods
- External data source integrations
- Custom processing utilities

**New Data Sources:**
- Additional document formats
- Real-time data streams
- External API integrations

**New Models:**
- Alternative language models
- Specialized embedding models
- Custom fine-tuned models

Technology Stack
----------------

**Core Framework:**
- CrewAI: Multi-agent orchestration
- LangChain: LLM integration and tooling
- FastAPI: API development (when needed)

**Data Layer:**
- Qdrant: Vector database
- Azure OpenAI: Language models
- Pandas: Data processing

**Infrastructure:**
- Docker: Containerization
- UV: Dependency management
- Streamlit: Web interface

**Monitoring:**
- Opik: AI operation monitoring
- Python logging: Application logs
- Custom metrics: Business intelligence

Next Steps
----------

To dive deeper into the architecture:

1. **Explore Crews**: :doc:`crews` - Detailed crew implementations
2. **Understand Tools**: :doc:`tools` - Core tool functionality
3. **Study Flows**: :doc:`flows` - Flow control and orchestration
4. **Review API**: :doc:`../api/index` - Code-level documentation