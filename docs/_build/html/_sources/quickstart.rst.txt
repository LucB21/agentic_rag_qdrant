Quick Start Guide
=================

This guide will help you get Agentic RAG up and running in just a few minutes.

Prerequisites
-------------

Before starting, make sure you have completed the :doc:`installation` and have:

- ✅ Python 3.10+ installed
- ✅ All dependencies installed via UV
- ✅ `.env` file configured with Azure OpenAI credentials
- ✅ Qdrant instance running (local or cloud)

Basic Usage
-----------

**1. Start with the Command Line Interface**

The simplest way to test the system:

.. code-block:: bash

   crewai run

This will start the default flow and process a sample query.

**2. Using the Streamlit Web Interface**

For an interactive experience:

.. code-block:: bash

   streamlit run streamlit_app.py

Then open your browser to `http://localhost:8501`.

**3. Programmatic Usage**

Create a Python script to use the RAG system:

.. code-block:: python

   from agentic_rag.main import RagFlow

   # Initialize the flow
   flow = RagFlow()
   
   # Set your question and sector
   flow.set_question_and_sector(
       question="What are the latest regulations in banking?",
       sector=["bancario"],
       skip_relevance_check=False
   )
   
   # Run the flow
   result = flow.kickoff()
   print(result)

First Query Example
-------------------

Let's walk through a complete example:

**1. Start the Streamlit Interface**

.. code-block:: bash

   streamlit run streamlit_app.py

**2. Enter Your Query**

In the web interface:

- **Question**: "What are the main risk management practices in banking?"
- **Sector**: Select "Banking" 
- **Skip Relevance Check**: Leave unchecked for full flow

**3. Review the Results**

The system will:

1. **Check Relevance**: Validate if the query is relevant to available knowledge
2. **Search Documents**: Find relevant information in the vector database
3. **Web Search** (if needed): Supplement with web search if local knowledge is insufficient
4. **Synthesize Response**: Combine information into a comprehensive answer

Understanding the Flow
----------------------

The Agentic RAG system follows this process:

.. mermaid::

   graph TD
       A[User Query] --> B[Check Crew]
       B --> C{Relevant?}
       C -->|Yes| D[RAG Search]
       C -->|No/Partial| E[Web Search Crew]
       D --> F[Synthesis Crew]
       E --> F
       F --> G[Final Response]

**Check Crew**
   Analyzes the query to determine:
   - Is it relevant to our knowledge base?
   - What sectors should be searched?
   - Should we proceed with RAG or web search?

**RAG Search**
   Performs hybrid search in Qdrant:
   - Semantic similarity using embeddings
   - Keyword matching for precise terms
   - MMR for diverse results

**Web Search Crew**
   Activated when:
   - Query is not fully relevant to local knowledge
   - Additional context is needed
   - Recent information is required

**Synthesis Crew**
   Combines all information sources:
   - Local RAG results
   - Web search results
   - Creates coherent, comprehensive response

Configuration Tips
------------------

**Adjusting Search Parameters**

Edit the RAG module configuration:

.. code-block:: python

   # In src/agentic_rag/tools/rag_module.py
   hybrid_search_params = {
       "semantic_weight": 0.7,  # Semantic vs keyword balance
       "mmr_diversity": 0.3,    # Result diversification
       "top_k": 10              # Number of results
   }

**Sector-Specific Searches**

The system supports multiple sectors:

- ``bancario`` - Banking and financial services
- ``energia`` - Energy and utilities
- Add more by updating the document indexing

**Relevance Threshold**

Adjust when web search is triggered:

.. code-block:: python

   # Lower threshold = more web searches
   # Higher threshold = more RAG-only responses
   relevance_threshold = 0.7

Common Use Cases
----------------

**1. Domain-Specific Research**

.. code-block:: python

   flow = RagFlow()
   flow.set_question_and_sector(
       question="Latest ESG regulations in energy sector",
       sector=["energia"],
       skip_relevance_check=False
   )
   result = flow.kickoff()

**2. Multi-Sector Analysis**

.. code-block:: python

   flow = RagFlow()
   flow.set_question_and_sector(
       question="Compare risk management across banking and energy",
       sector=["bancario", "energia"],
       skip_relevance_check=False
   )
   result = flow.kickoff()

**3. Quick RAG-Only Search**

.. code-block:: python

   flow = RagFlow()
   flow.set_question_and_sector(
       question="What is Basel III?",
       sector=["bancario"],
       skip_relevance_check=True  # Skip check, go directly to RAG
   )
   result = flow.kickoff()

Performance Optimization
------------------------

**For Better Speed:**

1. **Skip Relevance Check** for known relevant queries
2. **Use Specific Sectors** instead of searching all
3. **Reduce top_k** in search parameters
4. **Use Local Qdrant** instead of cloud for faster access

**For Better Quality:**

1. **Enable Full Flow** with relevance checking
2. **Allow Web Search** for comprehensive results
3. **Increase MMR diversity** for varied perspectives
4. **Use higher top_k** for more context

Troubleshooting Quick Issues
----------------------------

**No Results Returned**
   - Check if documents are indexed in Qdrant
   - Verify sector names match indexed collections
   - Try a broader or different query

**Slow Response**
   - Check Qdrant connection (local vs cloud)
   - Verify Azure OpenAI endpoint response time
   - Consider reducing search parameters

**Irrelevant Results**
   - Ensure proper document indexing
   - Check embedding model consistency
   - Adjust search weights in hybrid search

Next Steps
----------

Now that you have the basics working:

1. **Explore Advanced Configuration**: :doc:`configuration`
2. **Learn About Architecture**: :doc:`architecture/overview`
3. **Set Up Evaluation**: :doc:`evaluation/ragas`
4. **Review Examples**: :doc:`examples/basic_usage`
5. **Deploy to Production**: :doc:`deployment/production`

Need Help?
----------

- Check the **API Reference**: :doc:`api/index`
- Review **Common Issues**: :doc:`installation` troubleshooting section
- Explore **Usage Patterns**: :doc:`usage`