Agentic RAG Documentation
=========================

Welcome to **Agentic RAG**, an advanced multi-agent system for Retrieval-Augmented Generation powered by CrewAI, Qdrant vector database, and Azure OpenAI.

.. image:: https://img.shields.io/badge/Python-3.10%2B-blue
   :alt: Python Version

.. image:: https://img.shields.io/badge/CrewAI-Powered-green
   :alt: CrewAI Powered

.. image:: https://img.shields.io/badge/Vector%20DB-Qdrant-red
   :alt: Qdrant Vector Database

Overview
--------

Agentic RAG is a sophisticated multi-agent system that combines the power of:

* **CrewAI Framework**: Orchestrates multiple AI agents working collaboratively
* **Qdrant Vector Database**: High-performance vector similarity search with hybrid capabilities
* **Azure OpenAI**: Advanced language models for generation and embeddings
* **RAGAS Evaluation**: Comprehensive RAG system evaluation framework
* **Streamlit Interface**: User-friendly web interface for interaction

Key Features
------------

🤖 **Multi-Agent Architecture**
   - **Check Crew**: Validates query relevance and determines appropriate search strategy
   - **Web Search Crew**: Performs web searches when local knowledge is insufficient  
   - **Synthesis Crew**: Combines information from multiple sources into coherent responses

🔍 **Advanced Search Capabilities**
   - Hybrid search combining semantic and keyword matching
   - Maximum Marginal Relevance (MMR) for result diversification
   - Multi-sector document indexing (Banking, Energy, etc.)

📊 **Comprehensive Evaluation**
   - RAGAS framework integration for RAG evaluation
   - Metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy
   - Automated evaluation pipeline with CSV reporting

🌐 **User-Friendly Interface**
   - Streamlit web application for easy interaction
   - Real-time query processing and response generation
   - Sector-specific search options

Getting Started
---------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   quickstart
   configuration
   usage

.. toctree::
   :maxdepth: 2
   :caption: Architecture:

   architecture/overview
   architecture/crews
   architecture/tools
   architecture/flows

.. toctree::
   :maxdepth: 2
   :caption: Evaluation:

   evaluation/ragas
   evaluation/metrics
   evaluation/setup

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Deployment:

   deployment/docker
   deployment/production

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   examples/basic_usage
   examples/custom_crews
   examples/evaluation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


