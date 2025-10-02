Basic Usage Examples
====================

This guide provides practical examples of using Agentic RAG for different scenarios and use cases.

Getting Started
---------------

Before running examples, ensure you have:

- Completed the :doc:`../installation`
- Configured your `.env` file
- Started Qdrant (local or cloud)

Example 1: Simple Query
-----------------------

**Use Case**: Basic question answering with banking domain

.. code-block:: python

   from agentic_rag.main import RagFlow

   # Initialize the flow
   flow = RagFlow()

   # Set question and sector
   flow.set_question_and_sector(
       question="What is the minimum capital ratio under Basel III?",
       sector=["bancario"],
       skip_relevance_check=False
   )

   # Execute the flow
   result = flow.kickoff()
   
   # Display result
   print("Answer:", result.get('final_answer', 'No answer generated'))

**Expected Output**:

.. code-block:: text

   Answer: Under Basel III regulations, banks are required to maintain 
   a minimum capital adequacy ratio of 8%, with additional buffers 
   bringing the total requirement to approximately 10.5% in normal 
   conditions...

Example 2: Multi-Sector Analysis
---------------------------------

**Use Case**: Comparing regulations across different sectors

.. code-block:: python

   from agentic_rag.main import RagFlow

   # Initialize flow
   flow = RagFlow()

   # Multi-sector query
   flow.set_question_and_sector(
       question="Compare environmental regulations in banking vs energy sector",
       sector=["bancario", "energia"],
       skip_relevance_check=False
   )

   # Execute
   result = flow.kickoff()
   
   print("Comparative Analysis:")
   print(result.get('final_answer', 'Analysis not available'))

**Expected Behavior**:

1. **Check Crew** analyzes the comparative nature of the query
2. **RAG Tool** searches both banking and energy collections
3. **Synthesis Crew** creates a comparative analysis
4. Returns structured comparison of environmental regulations

Example 3: Web-Enhanced Search
------------------------------

**Use Case**: Recent information that may not be in the knowledge base

.. code-block:: python

   from agentic_rag.main import RagFlow

   # Initialize flow
   flow = RagFlow()

   # Query about recent developments
   flow.set_question_and_sector(
       question="What are the latest updates to ESG reporting requirements in 2024?",
       sector=["bancario", "energia"],
       skip_relevance_check=False
   )

   # Execute - this will likely trigger web search
   result = flow.kickoff()
   
   print("Latest Information:")
   print(result.get('final_answer', 'Information not available'))

**Expected Flow**:

1. **Check Crew** determines query needs recent information
2. **Web Search Crew** activated to find latest updates
3. **RAG Tool** provides foundational knowledge
4. **Synthesis Crew** combines local and web information

Example 4: Direct RAG Search
-----------------------------

**Use Case**: Skip relevance checking for known relevant queries

.. code-block:: python

   from agentic_rag.main import RagFlow

   # For known relevant queries, skip the check step
   flow = RagFlow()
   
   flow.set_question_and_sector(
       question="Explain the liquidity coverage ratio in Basel III",
       sector=["bancario"],
       skip_relevance_check=True  # Skip directly to RAG
   )

   result = flow.kickoff()
   print("Direct RAG Answer:", result.get('final_answer'))

**Benefits**:
- Faster response time
- Reduced API calls
- More predictable behavior

Example 5: Streaming Interface
------------------------------

**Use Case**: Real-time question answering with Streamlit

**Run the Streamlit App**:

.. code-block:: bash

   streamlit run streamlit_app.py

**Interactive Usage**:

1. **Open Browser**: Navigate to `http://localhost:8501`
2. **Enter Question**: "What are the key components of Solvency II?"
3. **Select Sector**: Choose "Banking" or "All Sectors"
4. **Advanced Options**: Configure search parameters
5. **Submit**: Get real-time response

**Streamlit Code Example**:

.. code-block:: python

   import streamlit as st
   from agentic_rag.main import RagFlow

   st.title("Agentic RAG Query Interface")

   # User inputs
   question = st.text_area("Enter your question:")
   sector = st.selectbox("Select sector:", ["bancario", "energia", "all"])
   skip_check = st.checkbox("Skip relevance check")

   if st.button("Submit Query"):
       with st.spinner("Processing..."):
           flow = RagFlow()
           flow.set_question_and_sector(
               question=question,
               sector=[sector] if sector != "all" else ["bancario", "energia"],
               skip_relevance_check=skip_check
           )
           
           result = flow.kickoff()
           
           st.success("Query completed!")
           st.write("**Answer:**")
           st.write(result.get('final_answer', 'No answer generated'))

Example 6: Batch Processing
---------------------------

**Use Case**: Processing multiple queries efficiently

.. code-block:: python

   import asyncio
   from agentic_rag.main import RagFlow

   async def process_batch_queries():
       """Process multiple queries in batch"""
       
       queries = [
           {
               "question": "What is operational risk in banking?",
               "sector": ["bancario"]
           },
           {
               "question": "How do smart grids work in energy distribution?", 
               "sector": ["energia"]
           },
           {
               "question": "Compare risk management frameworks",
               "sector": ["bancario", "energia"]
           }
       ]
       
       results = []
       
       for query in queries:
           flow = RagFlow()
           flow.set_question_and_sector(
               question=query["question"],
               sector=query["sector"],
               skip_relevance_check=False
           )
           
           result = flow.kickoff()
           results.append({
               "question": query["question"],
               "answer": result.get('final_answer'),
               "sector": query["sector"]
           })
           
           print(f"Processed: {query['question'][:50]}...")
       
       return results

   # Run batch processing
   if __name__ == "__main__":
       results = asyncio.run(process_batch_queries())
       
       # Display results
       for i, result in enumerate(results, 1):
           print(f"\n--- Query {i} ---")
           print(f"Q: {result['question']}")
           print(f"A: {result['answer'][:200]}...")

Example 7: Custom Tool Integration
----------------------------------

**Use Case**: Adding custom tools to the RAG workflow

.. code-block:: python

   from crewai.tools import tool
   from agentic_rag.main import RagFlow

   @tool("calculate_financial_ratio")
   def calculate_ratio(numerator: float, denominator: float) -> float:
       """Calculate financial ratios for analysis"""
       if denominator == 0:
           return 0
       return round(numerator / denominator, 4)

   # Custom crew with additional tools
   from agentic_rag.crews.synthesis_crew.synthesis_crew import SynthesisCrew

   class CustomSynthesisCrew(SynthesisCrew):
       def __init__(self):
           super().__init__()
           # Add custom tools to agents
           self.synthesis_agent.tools.append(calculate_ratio)

   # Use custom crew in flow
   class CustomRagFlow(RagFlow):
       def __init__(self):
           super().__init__()
           # Override with custom crew
           self.synthesis_crew = CustomSynthesisCrew()

   # Usage
   flow = CustomRagFlow()
   flow.set_question_and_sector(
       question="Calculate the capital adequacy ratio if Tier 1 capital is 120 million and risk-weighted assets are 1 billion",
       sector=["bancario"]
   )
   
   result = flow.kickoff()
   print(result.get('final_answer'))

Example 8: Error Handling
-------------------------

**Use Case**: Robust error handling in production

.. code-block:: python

   from agentic_rag.main import RagFlow
   import logging

   # Configure logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   def robust_query_processing(question, sector, max_retries=3):
       """Process query with error handling and retries"""
       
       for attempt in range(max_retries):
           try:
               flow = RagFlow()
               flow.set_question_and_sector(
                   question=question,
                   sector=sector,
                   skip_relevance_check=False
               )
               
               result = flow.kickoff()
               
               if result and result.get('final_answer'):
                   logger.info(f"Query successful on attempt {attempt + 1}")
                   return result
               else:
                   logger.warning(f"Empty result on attempt {attempt + 1}")
                   
           except Exception as e:
               logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
               
               if attempt == max_retries - 1:
                   # Last attempt failed
                   return {
                       "final_answer": f"Sorry, I couldn't process your query after {max_retries} attempts. Please try again later.",
                       "error": str(e)
                   }
               
               # Wait before retry
               import time
               time.sleep(2 ** attempt)  # Exponential backoff
       
       return None

   # Usage
   result = robust_query_processing(
       question="What are the main components of Basel III?",
       sector=["bancario"]
   )

   if result:
       print("Answer:", result['final_answer'])
       if 'error' in result:
           print("Error occurred:", result['error'])

Example 9: Performance Monitoring
---------------------------------

**Use Case**: Monitoring query performance and success rates

.. code-block:: python

   import time
   from agentic_rag.main import RagFlow
   import opik

   # Configure Opik for monitoring
   opik.configure(use_local=True)

   @opik.track(project_name="rag_performance_monitoring")
   def monitored_query(question, sector):
       """Query with performance monitoring"""
       
       start_time = time.time()
       
       try:
           flow = RagFlow()
           flow.set_question_and_sector(
               question=question,
               sector=sector,
               skip_relevance_check=False
           )
           
           result = flow.kickoff()
           
           # Log metrics
           processing_time = time.time() - start_time
           success = bool(result and result.get('final_answer'))
           
           # Track with Opik
           opik.log_metric("processing_time", processing_time)
           opik.log_metric("success", success)
           opik.log_metric("sector_count", len(sector))
           
           return {
               "answer": result.get('final_answer') if success else None,
               "processing_time": processing_time,
               "success": success
           }
           
       except Exception as e:
           processing_time = time.time() - start_time
           opik.log_metric("processing_time", processing_time)
           opik.log_metric("success", False)
           opik.log_metric("error", str(e))
           
           return {
               "answer": None,
               "processing_time": processing_time,
               "success": False,
               "error": str(e)
           }

   # Usage with monitoring
   result = monitored_query(
       question="Explain credit risk management in banking",
       sector=["bancario"]
   )

   print(f"Success: {result['success']}")
   print(f"Time: {result['processing_time']:.2f}s")
   if result['success']:
       print(f"Answer: {result['answer'][:100]}...")

Example 10: Configuration Testing
---------------------------------

**Use Case**: Testing different configuration parameters

.. code-block:: python

   from agentic_rag.main import RagFlow

   def test_configuration_impact():
       """Test different configurations and compare results"""
       
       test_question = "What are the main requirements of Basel III?"
       sector = ["bancario"]
       
       configurations = [
           {"skip_relevance_check": True, "name": "Direct RAG"},
           {"skip_relevance_check": False, "name": "Full Flow"},
       ]
       
       results = {}
       
       for config in configurations:
           start_time = time.time()
           
           flow = RagFlow()
           flow.set_question_and_sector(
               question=test_question,
               sector=sector,
               skip_relevance_check=config["skip_relevance_check"]
           )
           
           result = flow.kickoff()
           processing_time = time.time() - start_time
           
           results[config["name"]] = {
               "answer": result.get('final_answer'),
               "time": processing_time,
               "config": config
           }
       
       # Compare results
       print("Configuration Comparison:")
       print("=" * 50)
       
       for name, data in results.items():
           print(f"\n{name}:")
           print(f"Time: {data['time']:.2f}s")
           print(f"Answer Length: {len(data['answer']) if data['answer'] else 0} chars")
           print(f"Answer Preview: {data['answer'][:100] if data['answer'] else 'No answer'}...")

   # Run configuration test
   test_configuration_impact()

Best Practices
--------------

**1. Question Formulation**
   - Be specific and clear in your questions
   - Include relevant domain context
   - Avoid overly broad or ambiguous queries

**2. Sector Selection**
   - Choose specific sectors when possible
   - Use multiple sectors for comparative analysis
   - Consider "all" sectors only for general queries

**3. Performance Optimization**
   - Use `skip_relevance_check=True` for known relevant queries
   - Cache frequently asked questions
   - Monitor and log query performance

**4. Error Handling**
   - Implement retry mechanisms for critical applications
   - Log errors for debugging and improvement
   - Provide meaningful fallback responses

**5. Monitoring**
   - Track query success rates
   - Monitor response times
   - Analyze query patterns for optimization

Common Patterns
---------------

**Research Pattern**:
.. code-block:: python

   # Deep research on a specific topic
   flow.set_question_and_sector(
       question="Comprehensive analysis of [topic]",
       sector=["relevant_sector"],
       skip_relevance_check=False
   )

**Comparison Pattern**:
.. code-block:: python

   # Compare across sectors or topics
   flow.set_question_and_sector(
       question="Compare [topic A] vs [topic B]",
       sector=["sector1", "sector2"],
       skip_relevance_check=False
   )

**Quick Lookup Pattern**:
.. code-block:: python

   # Fast factual lookup
   flow.set_question_and_sector(
       question="What is [specific term]?",
       sector=["known_sector"],
       skip_relevance_check=True
   )

Next Steps
----------

After exploring these examples:

1. **Try Custom Crews**: :doc:`custom_crews` - Build specialized agent teams
2. **Evaluation Examples**: :doc:`evaluation` - Test your implementations
3. **Production Deployment**: :doc:`../deployment/production` - Scale your system
4. **Advanced Configuration**: :doc:`../configuration` - Fine-tune performance