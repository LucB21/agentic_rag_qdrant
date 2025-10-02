Installation Guide
==================

System Requirements
-------------------

* **Python**: 3.10 or higher (but less than 3.14)
* **Operating System**: Windows, macOS, or Linux
* **Memory**: Minimum 8GB RAM recommended
* **Storage**: At least 2GB free space for dependencies and vector storage

Prerequisites
-------------

Before installing Agentic RAG, ensure you have:

1. **Python Environment**: Python 3.10+ installed
2. **UV Package Manager**: For dependency management
3. **Azure OpenAI Access**: Valid API keys and endpoints
4. **Qdrant**: Either local installation or cloud access

Installing UV
-------------

UV is the recommended package manager for this project. Install it using:

.. code-block:: bash

   pip install uv

Quick Installation
------------------

1. **Clone the Repository**

   .. code-block:: bash

      git clone https://github.com/LucB21/agentic_rag_qdrant.git
      cd agentic_rag_qdrant

2. **Install Dependencies**

   .. code-block:: bash

      uv sync

   Or using CrewAI CLI:

   .. code-block:: bash

      crewai install

3. **Set Up Environment Variables**

   Create a `.env` file in the root directory:

   .. code-block:: bash

      # Azure OpenAI Configuration
      AZURE_OPENAI_API_KEY=your_api_key_here
      AZURE_OPENAI_ENDPOINT=your_endpoint_here
      AZURE_OPENAI_API_VERSION=2024-02-15-preview
      AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o
      AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME=text-embedding-ada-002

      # Qdrant Configuration
      QDRANT_URL=http://localhost:6333
      QDRANT_API_KEY=your_qdrant_api_key  # Optional for local instance

      # Optional: OpenAI API for web search fallback
      OPENAI_API_KEY=your_openai_api_key

4. **Verify Installation**

   .. code-block:: bash

      python -m agentic_rag --help

Development Installation
------------------------

For development and contribution:

1. **Clone with Development Dependencies**

   .. code-block:: bash

      git clone https://github.com/LucB21/agentic_rag_qdrant.git
      cd agentic_rag_qdrant

2. **Install in Development Mode**

   .. code-block:: bash

      uv sync --dev

3. **Install Pre-commit Hooks** (if available)

   .. code-block:: bash

      pre-commit install

Setting Up Qdrant
------------------

**Option 1: Local Qdrant with Docker**

.. code-block:: bash

   docker run -p 6333:6333 qdrant/qdrant

**Option 2: Qdrant Cloud**

1. Sign up at `Qdrant Cloud <https://cloud.qdrant.io/>`_
2. Create a cluster
3. Use the provided URL and API key in your `.env` file

**Option 3: Local Installation**

Follow the `official Qdrant installation guide <https://qdrant.tech/documentation/installation/>`_.

Azure OpenAI Setup
-------------------

1. **Create Azure OpenAI Resource**
   
   - Go to `Azure Portal <https://portal.azure.com/>`_
   - Create a new Azure OpenAI resource
   - Deploy the required models:
     - **Chat Model**: gpt-4o or gpt-4
     - **Embedding Model**: text-embedding-ada-002

2. **Get Credentials**
   
   - Copy the API key and endpoint from your Azure OpenAI resource
   - Note the deployment names for your models

3. **Update Environment Variables**
   
   Update your `.env` file with the Azure OpenAI credentials.

Troubleshooting
---------------

**Common Issues:**

1. **Import Errors**
   
   Ensure all dependencies are installed:
   
   .. code-block:: bash
   
      uv sync --reinstall

2. **Azure OpenAI Connection Issues**
   
   - Verify your API key and endpoint
   - Check if your models are deployed and accessible
   - Ensure your Azure subscription has sufficient quota

3. **Qdrant Connection Issues**
   
   - For local Qdrant: Ensure Docker container is running
   - For Qdrant Cloud: Verify URL and API key
   - Check firewall settings

4. **Memory Issues**
   
   - Increase system memory if possible
   - Reduce batch sizes in configuration
   - Consider using smaller embedding models

Verification
------------

After installation, verify everything works:

.. code-block:: bash

   # Test the basic flow
   crewai run

   # Test the Streamlit interface
   streamlit run streamlit_app.py

   # Test RAGAS evaluation
   cd test_ragas
   python rag_ragas_qdrant.py

Next Steps
----------

After successful installation:

1. Read the :doc:`quickstart` guide
2. Configure your system in :doc:`configuration`
3. Explore :doc:`usage` examples
4. Set up :doc:`evaluation/ragas` for system assessment