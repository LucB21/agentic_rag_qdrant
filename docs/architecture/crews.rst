Agent Crews Architecture
========================

This document provides detailed information about the multi-agent crews that power Agentic RAG.

Crew Overview
-------------

Agentic RAG uses three specialized crews, each designed for specific tasks:

1. **Check Crew**: Query relevance analysis and routing decisions
2. **Web Search Crew**: External information retrieval when needed
3. **Synthesis Crew**: Information combination and response generation

Each crew consists of one or more AI agents working collaboratively to achieve their objectives.

Check Crew
----------

**Purpose**: Determines whether a user query is relevant to the local knowledge base and decides on the appropriate processing strategy.

Architecture
~~~~~~~~~~~~

.. code-block:: python

   class CheckCrew:
       def __init__(self):
           self.relevance_checker = Agent(
               role="Question Relevance Analyst",
               goal="Determine query relevance to knowledge domains",
               backstory="Expert analyst for domain-specific queries",
               llm=azure_openai_gpt4
           )

Agent Configuration
~~~~~~~~~~~~~~~~~~

**Relevance Checker Agent**:

- **Role**: Question Relevance Analyst
- **Goal**: Determine if user questions are relevant to available knowledge domains
- **Backstory**: Expert analyst who evaluates whether user queries can be effectively answered using available domain-specific knowledge bases
- **Tools**: Query analysis tools, domain classification
- **LLM**: Azure OpenAI GPT-4o

Task Definition
~~~~~~~~~~~~~~

.. code-block:: yaml

   check_relevance:
     description: >
       Analyze the user's question: "{question}" 
       Determine if it's relevant to banking, energy, or other available domains.
       Consider whether the question requires:
       1. Domain-specific knowledge from our documents
       2. Recent information not in our knowledge base
       3. General knowledge that might need web search
     
     expected_output: >
       A structured analysis containing:
       - Relevance score (0.0 to 1.0)
       - Recommended sectors to search
       - Processing strategy: 'rag_only', 'web_enhanced', or 'web_primary'
       - Confidence level in the recommendation

Decision Logic
~~~~~~~~~~~~~

The Check Crew uses the following decision tree:

.. code-block:: python

   def analyze_query_relevance(question, available_sectors):
       analysis = {
           "relevance_score": 0.0,
           "recommended_sectors": [],
           "strategy": "web_primary",
           "confidence": 0.0
       }
       
       # Domain keyword analysis
       domain_keywords = extract_domain_keywords(question)
       sector_matches = match_sectors(domain_keywords, available_sectors)
       
       if sector_matches:
           analysis["relevance_score"] = calculate_relevance_score(sector_matches)
           analysis["recommended_sectors"] = sector_matches
           
           if analysis["relevance_score"] >= 0.8:
               analysis["strategy"] = "rag_only"
           elif analysis["relevance_score"] >= 0.5:
               analysis["strategy"] = "web_enhanced"
           else:
               analysis["strategy"] = "web_primary"
       
       return analysis

Example Scenarios
~~~~~~~~~~~~~~~~

**High Relevance (RAG Only)**:

.. code-block:: text

   Question: "What are the capital adequacy requirements under Basel III?"
   Analysis:
   - Relevance Score: 0.95
   - Sectors: ["bancario"]
   - Strategy: "rag_only"
   - Confidence: 0.92

**Medium Relevance (Web Enhanced)**:

.. code-block:: text

   Question: "How do recent ESG regulations impact banking operations?"
   Analysis:
   - Relevance Score: 0.65
   - Sectors: ["bancario"]
   - Strategy: "web_enhanced"
   - Confidence: 0.78

**Low Relevance (Web Primary)**:

.. code-block:: text

   Question: "What's the weather forecast for tomorrow?"
   Analysis:
   - Relevance Score: 0.1
   - Sectors: []
   - Strategy: "web_primary"
   - Confidence: 0.95

Web Search Crew
---------------

**Purpose**: Retrieves current and comprehensive information from web sources when local knowledge is insufficient.

Architecture
~~~~~~~~~~~~

.. code-block:: python

   class WebSearchCrew:
       def __init__(self):
           self.web_researcher = Agent(
               role="Web Research Specialist",
               goal="Find comprehensive and current information through web searches",
               backstory="Skilled researcher with expertise in finding relevant information",
               tools=[web_search_tool, url_scraper_tool],
               llm=azure_openai_gpt4
           )

Agent Configuration
~~~~~~~~~~~~~~~~~~

**Web Researcher Agent**:

- **Role**: Web Research Specialist
- **Goal**: Find comprehensive and current information through web searches
- **Backstory**: Skilled researcher who excels at finding relevant, current information from web sources
- **Tools**: Web search, URL scraping, content extraction
- **LLM**: Azure OpenAI GPT-4o

Available Tools
~~~~~~~~~~~~~~

**1. Web Search Tool**:

.. code-block:: python

   @tool("web_search")
   def web_search_tool(query: str, num_results: int = 5) -> List[str]:
       """Search the web for information related to the query"""
       # Implementation using DuckDuckGo, Bing, or Google
       pass

**2. URL Scraper Tool**:

.. code-block:: python

   @tool("scrape_url")
   def url_scraper_tool(url: str) -> str:
       """Extract content from a specific URL"""
       # Implementation for content extraction
       pass

**3. Content Summarizer Tool**:

.. code-block:: python

   @tool("summarize_content")
   def content_summarizer_tool(content: str, focus: str) -> str:
       """Summarize web content focusing on specific aspects"""
       # Implementation for content summarization
       pass

Task Configuration
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   web_search_task:
     description: >
       Search the web for information related to: "{question}"
       Focus on finding:
       1. Recent developments and updates
       2. Expert opinions and analysis
       3. Regulatory changes or announcements
       4. Industry trends and statistics
     
     expected_output: >
       A comprehensive summary of web findings including:
       - Key information points
       - Source URLs and credibility assessment
       - Relevance to the original question
       - Recency of information

Search Strategy
~~~~~~~~~~~~~~

.. code-block:: python

   class WebSearchStrategy:
       def execute_search(self, question, context=None):
           # Step 1: Generate search queries
           search_queries = self.generate_search_queries(question, context)
           
           # Step 2: Execute searches
           search_results = []
           for query in search_queries:
               results = self.web_search_tool(query, num_results=3)
               search_results.extend(results)
           
           # Step 3: Filter and rank results
           filtered_results = self.filter_results(search_results, question)
           
           # Step 4: Extract and summarize content
           summaries = []
           for result in filtered_results[:5]:  # Top 5 results
               content = self.url_scraper_tool(result['url'])
               summary = self.content_summarizer_tool(content, question)
               summaries.append(summary)
           
           return summaries

Synthesis Crew
--------------

**Purpose**: Combines information from multiple sources (RAG results and web search) to create comprehensive, coherent responses.

Architecture
~~~~~~~~~~~

.. code-block:: python

   class SynthesisCrew:
       def __init__(self):
           self.synthesis_agent = Agent(
               role="Information Synthesis Expert",
               goal="Combine information from multiple sources into comprehensive responses",
               backstory="Expert at synthesizing information from various sources",
               tools=[citation_tool, fact_checker_tool],
               llm=azure_openai_gpt4
           )

Agent Configuration
~~~~~~~~~~~~~~~~~~

**Synthesis Agent**:

- **Role**: Information Synthesis Expert
- **Goal**: Combine information from multiple sources into comprehensive responses
- **Backstory**: Expert at synthesizing information from various sources to create coherent, comprehensive, and accurate responses
- **Tools**: Citation generation, fact checking, content structuring
- **LLM**: Azure OpenAI GPT-4o

Synthesis Process
~~~~~~~~~~~~~~~~

.. code-block:: python

   class SynthesisProcess:
       def synthesize_response(self, question, rag_results, web_results=None):
           # Step 1: Analyze information sources
           source_analysis = self.analyze_sources(rag_results, web_results)
           
           # Step 2: Identify key themes and topics
           themes = self.extract_themes(source_analysis)
           
           # Step 3: Resolve conflicts and contradictions
           resolved_info = self.resolve_conflicts(source_analysis)
           
           # Step 4: Structure response
           response_structure = self.create_response_structure(themes, resolved_info)
           
           # Step 5: Generate final response
           final_response = self.generate_response(question, response_structure)
           
           return final_response

Task Configuration
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   synthesis_task:
     description: >
       Synthesize information from multiple sources to answer: "{question}"
       
       Available information:
       - RAG Results: {rag_context}
       - Web Search Results: {web_context}
       
       Create a comprehensive response that:
       1. Directly answers the question
       2. Provides supporting details and context
       3. Integrates information from all sources
       4. Identifies and resolves any contradictions
       5. Includes proper citations
     
     expected_output: >
       A well-structured response containing:
       - Clear answer to the question
       - Supporting evidence and details
       - Proper source attribution
       - Confidence indicators
       - Suggestions for further reading

Response Structure
~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ResponseFormatter:
       def format_response(self, content, sources):
           response = {
               "main_answer": self.extract_main_answer(content),
               "supporting_details": self.extract_details(content),
               "sources": self.format_sources(sources),
               "confidence_score": self.calculate_confidence(content, sources),
               "limitations": self.identify_limitations(content, sources)
           }
           return response

Crew Interaction Patterns
-------------------------

Sequential Processing
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class SequentialFlow:
       def process_query(self, question, sector):
           # Step 1: Check relevance
           check_result = self.check_crew.process(question)
           
           # Step 2: Based on strategy, choose next step
           if check_result.strategy == "rag_only":
               rag_results = self.rag_tool.search(question, sector)
               final_response = self.synthesis_crew.process(question, rag_results)
           
           elif check_result.strategy == "web_enhanced":
               rag_results = self.rag_tool.search(question, sector)
               web_results = self.web_search_crew.process(question)
               final_response = self.synthesis_crew.process(question, rag_results, web_results)
           
           else:  # web_primary
               web_results = self.web_search_crew.process(question)
               final_response = self.synthesis_crew.process(question, None, web_results)
           
           return final_response

Parallel Processing
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio

   class ParallelFlow:
       async def process_query_parallel(self, question, sector):
           # Start check and RAG search in parallel
           check_task = asyncio.create_task(self.check_crew.process_async(question))
           rag_task = asyncio.create_task(self.rag_tool.search_async(question, sector))
           
           # Wait for check result to determine if web search is needed
           check_result = await check_task
           
           if check_result.strategy in ["web_enhanced", "web_primary"]:
               web_task = asyncio.create_task(self.web_search_crew.process_async(question))
               rag_results, web_results = await asyncio.gather(rag_task, web_task)
           else:
               rag_results = await rag_task
               web_results = None
           
           # Synthesize final response
           final_response = await self.synthesis_crew.process_async(
               question, rag_results, web_results
           )
           
           return final_response

Error Handling and Fallbacks
----------------------------

Crew-Level Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class RobustCrewExecution:
       def execute_with_fallback(self, crew, task, max_retries=3):
           for attempt in range(max_retries):
               try:
                   result = crew.kickoff(task)
                   if self.validate_result(result):
                       return result
                   else:
                       raise ValueError("Invalid result format")
               
               except Exception as e:
                   if attempt == max_retries - 1:
                       return self.get_fallback_response(task, e)
                   
                   # Wait before retry with exponential backoff
                   time.sleep(2 ** attempt)
           
           return self.get_fallback_response(task, "Max retries exceeded")

Fallback Strategies
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class FallbackManager:
       def get_fallback_response(self, question, error_context):
           if "check_crew" in error_context:
               # Default to web-enhanced strategy
               return {"strategy": "web_enhanced", "confidence": 0.5}
           
           elif "web_search_crew" in error_context:
               # Fall back to RAG-only
               return self.rag_tool.search(question, ["all"])
           
           elif "synthesis_crew" in error_context:
               # Return simple formatted response
               return self.simple_response_formatter(question, raw_results)
           
           else:
               return {"error": "Unable to process query", "suggestion": "Please try again"}

Performance Optimization
------------------------

Crew Caching
~~~~~~~~~~~~

.. code-block:: python

   from functools import lru_cache
   import hashlib

   class CrewCache:
       @lru_cache(maxsize=100)
       def cached_check_result(self, question_hash):
           # Cache check crew results for similar questions
           pass
       
       @lru_cache(maxsize=200)
       def cached_web_search(self, query_hash):
           # Cache web search results
           pass

Async Execution
~~~~~~~~~~~~~~

.. code-block:: python

   class AsyncCrewManager:
       async def execute_crews_async(self, question, sector):
           # Use asyncio for concurrent execution where possible
           tasks = []
           
           if self.should_run_check():
               tasks.append(self.check_crew.process_async(question))
           
           if self.should_run_rag():
               tasks.append(self.rag_tool.search_async(question, sector))
           
           results = await asyncio.gather(*tasks, return_exceptions=True)
           return self.process_results(results)

Monitoring and Observability
----------------------------

Crew Performance Metrics
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import opik
   from datetime import datetime

   class CrewMonitoring:
       @opik.track(project_name="crew_performance")
       def monitor_crew_execution(self, crew_name, task, execution_time):
           opik.log_metric(f"{crew_name}_execution_time", execution_time)
           opik.log_metric(f"{crew_name}_success_rate", 1.0)
           
       def log_crew_failure(self, crew_name, error):
           opik.log_metric(f"{crew_name}_success_rate", 0.0)
           opik.log_event(f"{crew_name}_error", {"error": str(error)})

Custom Crew Development
----------------------

Creating Custom Crews
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CustomAnalysisCrew:
       def __init__(self):
           self.analyst = Agent(
               role="Domain Expert Analyst",
               goal="Provide specialized analysis for specific domains",
               backstory="Expert analyst with deep domain knowledge",
               tools=[custom_analysis_tool],
               llm=azure_openai_gpt4
           )
           
           self.task = Task(
               description="Analyze {input} using domain expertise",
               expected_output="Detailed analysis with recommendations",
               agent=self.analyst
           )
       
       def analyze(self, input_data):
           return self.crew.kickoff({"input": input_data})

Extending Existing Crews
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ExtendedSynthesisCrew(SynthesisCrew):
       def __init__(self):
           super().__init__()
           
           # Add additional agent
           self.fact_checker = Agent(
               role="Fact Verification Specialist",
               goal="Verify accuracy of synthesized information",
               backstory="Expert at fact-checking and verification",
               tools=[fact_check_tool, source_verification_tool],
               llm=azure_openai_gpt4
           )
           
           # Add fact-checking task
           self.fact_check_task = Task(
               description="Verify facts in the synthesized response",
               expected_output="Fact-checked response with confidence scores",
               agent=self.fact_checker
           )

Next Steps
----------

To learn more about crews:

1. **Explore Tools**: :doc:`tools` - Understanding the tools available to crews
2. **Study Flows**: :doc:`flows` - How crews are orchestrated in flows
3. **API Reference**: :doc:`../api/index` - Detailed code documentation
4. **Custom Development**: Create your own specialized crews