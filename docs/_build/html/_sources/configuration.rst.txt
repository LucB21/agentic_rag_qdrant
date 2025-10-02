Configuration Guide
===================

This guide covers all configuration options for Agentic RAG, from basic setup to advanced customization.

Environment Variables
----------------------

All core configuration is handled through environment variables in your `.env` file.

Azure OpenAI Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Required: Azure OpenAI API Configuration
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   
   # Model Deployment Names
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o
   AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=text-embedding-ada-002
   
   # Optional: Model Parameters
   AZURE_OPENAI_TEMPERATURE=0.1
   AZURE_OPENAI_MAX_TOKENS=4000

Qdrant Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Local Qdrant Instance
   QDRANT_URL=http://localhost:6333
   
   # Or Qdrant Cloud
   QDRANT_URL=https://your-cluster.qdrant.app
   QDRANT_API_KEY=your_qdrant_api_key
   
   # Collection Settings
   QDRANT_COLLECTION_NAME=agentic_rag_docs
   QDRANT_VECTOR_SIZE=1536  # For text-embedding-ada-002

Web Search Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Optional: OpenAI for web search fallback
   OPENAI_API_KEY=your_openai_api_key
   
   # Search Engine Configuration
   SEARCH_ENGINE=duckduckgo  # or bing, google
   MAX_SEARCH_RESULTS=10

Monitoring and Logging
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Opik Monitoring (optional)
   OPIK_PROJECT_NAME=agentic_rag_project
   OPIK_USE_LOCAL=true
   
   # Logging Level
   LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

Crew Configuration
------------------

Each crew can be configured through YAML files in the `config/` directories.

Check Crew Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Located at `src/agentic_rag/crews/check_crew/config/agents.yaml`:

.. code-block:: yaml

   relevance_checker:
     role: >
       Question Relevance Analyst
     goal: >
       Determine if user questions are relevant to available knowledge domains
     backstory: >
       You are an expert analyst who evaluates whether user queries can be 
       effectively answered using available domain-specific knowledge bases.
     llm: azure_openai_gpt4
     max_iter: 3
     temperature: 0.1

And `src/agentic_rag/crews/check_crew/config/tasks.yaml`:

.. code-block:: yaml

   check_relevance:
     description: >
       Analyze the user's question: "{question}" 
       Determine if it's relevant to banking, energy, or other available domains.
     expected_output: >
       A structured analysis indicating relevance score, recommended sectors,
       and whether to proceed with RAG search or web search.

Web Search Crew Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Located at `src/agentic_rag/crews/web_search_crew/config/agents.yaml`:

.. code-block:: yaml

   web_researcher:
     role: >
       Web Research Specialist
     goal: >
       Find comprehensive and current information through web searches
     backstory: >
       You are a skilled researcher who excels at finding relevant, 
       current information from web sources.
     llm: azure_openai_gpt4
     max_execution_time: 180
     max_iter: 5

Synthesis Crew Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Located at `src/agentic_rag/crews/synthesis_crew/config/agents.yaml`:

.. code-block:: yaml

   synthesis_agent:
     role: >
       Information Synthesis Expert
     goal: >
       Combine information from multiple sources into comprehensive responses
     backstory: >
       You are an expert at synthesizing information from various sources
       to create coherent, comprehensive, and accurate responses.
     llm: azure_openai_gpt4
     temperature: 0.2

RAG Tool Configuration
----------------------

The RAG tool can be configured in `src/agentic_rag/tools/rag_module.py`:

Vector Search Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Hybrid Search Configuration
   HYBRID_SEARCH_CONFIG = {
       "semantic_weight": 0.7,        # Weight for semantic search (0.0-1.0)
       "keyword_weight": 0.3,         # Weight for keyword search (0.0-1.0)
       "top_k": 20,                   # Initial results to retrieve
       "mmr_diversity_factor": 0.3,   # Diversity in MMR (0.0-1.0)
       "final_k": 5                   # Final results after MMR
   }

   # Qdrant Search Configuration
   QDRANT_CONFIG = {
       "search_params": {
           "hnsw_ef": 128,            # HNSW search parameter
           "exact": False             # Use approximate search
       },
       "score_threshold": 0.3         # Minimum relevance score
   }

Document Processing Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Document Chunking
   CHUNK_CONFIG = {
       "chunk_size": 1000,           # Characters per chunk
       "chunk_overlap": 200,         # Overlap between chunks
       "separators": ["\n\n", "\n", ". ", "! ", "? "]
   }

   # Supported File Types
   SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".docx"]

Streamlit Interface Configuration
---------------------------------

Configure the web interface in `streamlit_app.py`:

UI Configuration
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Page Configuration
   st.set_page_config(
       page_title="Agentic RAG System",
       page_icon="🤖",
       layout="wide",
       initial_sidebar_state="expanded"
   )

   # Available Sectors
   AVAILABLE_SECTORS = {
       "bancario": "Banking & Finance",
       "energia": "Energy & Utilities",
       "all": "All Sectors"
   }

   # Interface Settings
   UI_CONFIG = {
       "max_question_length": 500,
       "default_sector": "all",
       "show_advanced_options": True,
       "enable_history": True
   }

RAGAS Evaluation Configuration
------------------------------

Configure evaluation in `test_ragas/rag_ragas_qdrant.py`:

Evaluation Metrics
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # RAGAS Metrics Configuration
   RAGAS_METRICS = [
       context_precision,          # How precise is the retrieved context
       context_recall,            # How much relevant context is retrieved  
       faithfulness,              # How faithful is the answer to context
       answer_relevancy,          # How relevant is the answer to question
       answer_correctness         # Overall correctness of the answer
   ]

   # Evaluation Parameters
   EVAL_CONFIG = {
       "test_size": 20,           # Number of test questions
       "batch_size": 5,           # Evaluation batch size
       "max_retries": 3,          # Retries for failed evaluations
       "timeout": 60              # Timeout per evaluation (seconds)
   }

Dataset Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test Dataset
   EVALUATION_QUESTIONS = [
       {
           "question": "What are Basel III requirements?",
           "sector": "bancario",
           "expected_topics": ["basel", "capital", "requirements"]
       },
       {
           "question": "How does renewable energy affect grid stability?",
           "sector": "energia", 
           "expected_topics": ["renewable", "grid", "stability"]
       }
   ]

Advanced Configuration
----------------------

Performance Tuning
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Memory Management
   MEMORY_CONFIG = {
       "max_concurrent_requests": 5,
       "request_timeout": 300,
       "cache_size": 1000,
       "garbage_collection_interval": 100
   }

   # Qdrant Performance
   QDRANT_PERFORMANCE = {
       "prefer_grpc": True,          # Use gRPC for better performance
       "connection_pool_size": 10,   # Connection pool size
       "timeout": 30,                # Request timeout
       "retries": 3                  # Number of retries
   }

Security Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # API Security
   SECURITY_CONFIG = {
       "rate_limit": "100/hour",     # Rate limiting
       "max_request_size": "10MB",   # Maximum request size
       "enable_cors": True,          # Enable CORS
       "allowed_origins": ["*"]      # Allowed origins
   }

   # Data Privacy
   PRIVACY_CONFIG = {
       "log_queries": False,         # Log user queries
       "anonymize_logs": True,       # Anonymize logged data
       "data_retention_days": 30     # Data retention period
   }

Development Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Debug Settings
   DEBUG_CONFIG = {
       "verbose_logging": True,
       "profile_performance": False,
       "save_intermediate_results": True,
       "debug_embeddings": False
   }

Configuration Validation
-------------------------

Use the built-in configuration validator:

.. code-block:: python

   from agentic_rag.config_validator import validate_config
   
   # Validate current configuration
   is_valid, errors = validate_config()
   
   if not is_valid:
       for error in errors:
           print(f"Configuration Error: {error}")

Environment-Specific Configurations
------------------------------------

Development Environment
~~~~~~~~~~~~~~~~~~~~~~~

Create `.env.development`:

.. code-block:: bash

   # Development settings
   LOG_LEVEL=DEBUG
   AZURE_OPENAI_TEMPERATURE=0.0
   QDRANT_URL=http://localhost:6333
   OPIK_USE_LOCAL=true

Production Environment
~~~~~~~~~~~~~~~~~~~~~~

Create `.env.production`:

.. code-block:: bash

   # Production settings
   LOG_LEVEL=WARNING
   AZURE_OPENAI_TEMPERATURE=0.1
   QDRANT_URL=https://your-production-cluster.qdrant.app
   QDRANT_API_KEY=your_production_api_key
   OPIK_USE_LOCAL=false

Testing Environment
~~~~~~~~~~~~~~~~~~~

Create `.env.testing`:

.. code-block:: bash

   # Testing settings
   LOG_LEVEL=ERROR
   AZURE_OPENAI_TEMPERATURE=0.0
   QDRANT_URL=http://localhost:6333
   RAGAS_TEST_SIZE=5

Configuration Best Practices
-----------------------------

1. **Security**
   - Never commit `.env` files to version control
   - Use strong API keys and rotate them regularly
   - Limit API access to necessary resources only

2. **Performance**
   - Use local Qdrant for development
   - Configure appropriate timeout values
   - Monitor and adjust chunk sizes based on your documents

3. **Reliability**
   - Set appropriate retry policies
   - Configure proper error handling
   - Use health checks for external services

4. **Monitoring**
   - Enable logging for production
   - Use Opik for performance monitoring
   - Set up alerts for API rate limits

5. **Cost Optimization**
   - Monitor Azure OpenAI usage
   - Adjust model parameters to balance cost/quality
   - Use caching for frequently accessed results

Next Steps
----------

After configuring your system:

1. Test configuration with :doc:`quickstart`
2. Understand the :doc:`architecture/overview`
3. Set up :doc:`evaluation/ragas`
4. Deploy using :doc:`deployment/production`