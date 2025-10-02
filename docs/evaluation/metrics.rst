Evaluation Metrics
==================

This page provides detailed information about each RAGAS evaluation metric used in Agentic RAG.

Overview
--------

RAGAS provides five key metrics to evaluate different aspects of RAG system performance:

.. list-table:: RAGAS Metrics Summary
   :header-rows: 1
   :widths: 20 30 20 30

   * - Metric
     - What it Measures
     - Range
     - Good Score
   * - Context Precision
     - Quality of retrieved context
     - 0.0 - 1.0
     - > 0.70
   * - Context Recall
     - Completeness of retrieved context
     - 0.0 - 1.0
     - > 0.65
   * - Faithfulness
     - Answer alignment with context
     - 0.0 - 1.0
     - > 0.75
   * - Answer Relevancy
     - Answer relevance to question
     - 0.0 - 1.0
     - > 0.80
   * - Answer Correctness
     - Overall answer accuracy
     - 0.0 - 1.0
     - > 0.70

Detailed Metric Descriptions
-----------------------------

Context Precision
~~~~~~~~~~~~~~~~~

**Formula**: Number of relevant chunks / Total retrieved chunks

**Calculation Process**:

1. For each retrieved context chunk
2. LLM evaluates: "Is this chunk relevant to answering the question?"
3. Count relevant chunks vs total chunks
4. Calculate ratio

**Example**:

.. code-block:: python

   Question: "What are Basel III capital requirements?"
   
   Retrieved Contexts:
   1. "Basel III requires 8% minimum capital ratio..." ✓ Relevant
   2. "Basel III timeline for implementation..." ✓ Relevant  
   3. "Weather forecast for tomorrow..." ✗ Not Relevant
   4. "Recipe for chocolate cake..." ✗ Not Relevant
   
   Context Precision = 2/4 = 0.50

**Improving Context Precision**:

- Better query preprocessing
- Improved semantic search weights
- Enhanced document chunking
- Stricter relevance thresholds

Context Recall
~~~~~~~~~~~~~~

**Formula**: Retrieved relevant information / Total relevant information in ground truth

**Calculation Process**:

1. Extract key facts from ground truth answer
2. Check which facts are present in retrieved contexts
3. Calculate coverage ratio

**Example**:

.. code-block:: python

   Question: "What are Basel III main components?"
   
   Ground Truth Key Facts:
   1. Capital requirements (8% minimum)
   2. Liquidity coverage ratio (LCR)
   3. Net stable funding ratio (NSFR)
   4. Leverage ratio (3% minimum)
   
   Retrieved Context Contains:
   1. Capital requirements ✓
   2. Liquidity coverage ratio ✓
   3. Net stable funding ratio ✗ (missing)
   4. Leverage ratio ✗ (missing)
   
   Context Recall = 2/4 = 0.50

**Improving Context Recall**:

- Increase top_k in retrieval
- Improve document indexing coverage
- Use multiple retrieval strategies
- Enhance query expansion

Faithfulness
~~~~~~~~~~~~

**Formula**: Number of supported claims / Total claims in answer

**Calculation Process**:

1. Extract factual claims from generated answer
2. For each claim, check if it's supported by retrieved context
3. Calculate ratio of supported claims

**Example**:

.. code-block:: python

   Generated Answer: "Basel III requires banks to maintain 8% capital ratio 
   and was implemented in 2015. It also requires 15% liquidity coverage."
   
   Claims Analysis:
   1. "8% capital ratio" ✓ Supported by context
   2. "implemented in 2015" ✓ Supported by context
   3. "15% liquidity coverage" ✗ Context says 100%, not 15%
   
   Faithfulness = 2/3 = 0.67

**Improving Faithfulness**:

- Better prompt engineering
- Lower temperature settings
- Explicit context citation requirements
- Enhanced context formatting

Answer Relevancy
~~~~~~~~~~~~~~~~

**Formula**: Cosine similarity between question and generated question from answer

**Calculation Process**:

1. Generate possible questions that the answer could address
2. Calculate semantic similarity with original question
3. Average similarity scores

**Example**:

.. code-block:: python

   Original Question: "What are Basel III capital requirements?"
   
   Generated Answer: "Basel III capital requirements include 8% minimum 
   capital adequacy ratio with additional buffers..."
   
   Reverse-Generated Questions from Answer:
   1. "What are the capital requirements under Basel III?" (high similarity)
   2. "How much capital must banks hold under Basel III?" (high similarity)
   3. "What is the minimum capital ratio?" (medium similarity)
   
   Average Similarity = 0.87

**Improving Answer Relevancy**:

- Better question understanding
- Focused answer generation
- Reduced hallucination
- Improved context selection

Answer Correctness
~~~~~~~~~~~~~~~~~~

**Formula**: Combination of semantic similarity and factual accuracy with ground truth

**Calculation Process**:

1. Calculate semantic similarity between answer and ground truth
2. Assess factual accuracy using LLM evaluation
3. Combine scores with weighting

**Example**:

.. code-block:: python

   Ground Truth: "Basel III requires 8% minimum capital ratio with 
   additional conservation buffer of 2.5%"
   
   Generated Answer: "Basel III mandates 8% capital adequacy ratio 
   and includes conservation buffers"
   
   Semantic Similarity: 0.85 (high conceptual overlap)
   Factual Accuracy: 0.80 (missing specific 2.5% detail)
   
   Answer Correctness = (0.85 + 0.80) / 2 = 0.825

**Improving Answer Correctness**:

- Higher quality ground truth data
- Better training data for evaluation
- Improved synthesis prompts
- Enhanced fact verification

Metric Relationships
--------------------

Understanding how metrics relate to each other:

**High Precision + Low Recall**:
- System is conservative, retrieves only very relevant content
- Answers may be incomplete but accurate
- Good for high-stakes scenarios

**Low Precision + High Recall**:
- System retrieves broadly, including some irrelevant content
- More comprehensive but potentially noisy answers
- Risk of confusion or misinformation

**High Faithfulness + Low Relevancy**:
- Answer accurately reflects context but doesn't address question
- Possible query understanding issues
- Context may be relevant but answer generation failed

**Low Faithfulness + High Relevancy**:
- Answer addresses the question but introduces unsupported claims
- Hallucination or over-generation issues
- Context retrieval successful but synthesis failed

Interpretation Guidelines
-------------------------

Score Ranges and Actions
~~~~~~~~~~~~~~~~~~~~~~~~

**0.9 - 1.0 (Excellent)**:
- System performing at production level
- Minor optimizations may still be beneficial
- Focus on consistency and edge cases

**0.7 - 0.9 (Good)**:
- System ready for most use cases
- Identify specific failure patterns
- Targeted improvements possible

**0.5 - 0.7 (Fair)**:
- System needs improvement before production
- Systematic issues likely present
- Review architecture and configuration

**0.0 - 0.5 (Poor)**:
- Significant problems requiring attention
- Check data quality and system configuration
- Consider fundamental changes

Debugging with Metrics
~~~~~~~~~~~~~~~~~~~~~~

**Low Context Precision**:
- Check document relevance in index
- Adjust search parameters
- Improve query preprocessing

**Low Context Recall**:
- Increase retrieval count
- Check document coverage
- Improve chunking strategy

**Low Faithfulness**:
- Review prompt templates
- Check for hallucination patterns
- Adjust generation parameters

**Low Answer Relevancy**:
- Improve question routing
- Check agent understanding
- Review task definitions

**Low Answer Correctness**:
- Verify ground truth quality
- Check knowledge base accuracy
- Improve synthesis quality

Metric Calculation Code
-----------------------

Example implementation for custom metrics:

.. code-block:: python

   from ragas import evaluate
   from ragas.metrics import (
       context_precision,
       context_recall, 
       faithfulness,
       answer_relevancy,
       answer_correctness
   )

   def calculate_ragas_metrics(dataset):
       """Calculate all RAGAS metrics for a dataset"""
       
       # Define metrics to evaluate
       metrics = [
           context_precision,
           context_recall,
           faithfulness,
           answer_relevancy,
           answer_correctness
       ]
       
       # Run evaluation
       result = evaluate(
           dataset=dataset,
           metrics=metrics,
           llm=azure_openai_llm,
           embeddings=azure_openai_embeddings
       )
       
       return result

   def analyze_metric_correlations(results_df):
       """Analyze correlations between different metrics"""
       
       import pandas as pd
       import matplotlib.pyplot as plt
       import seaborn as sns
       
       # Calculate correlation matrix
       metrics = ['context_precision', 'context_recall', 
                 'faithfulness', 'answer_relevancy', 'answer_correctness']
       
       correlation_matrix = results_df[metrics].corr()
       
       # Visualize correlations
       plt.figure(figsize=(10, 8))
       sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
       plt.title('RAGAS Metrics Correlation Matrix')
       plt.tight_layout()
       plt.show()
       
       return correlation_matrix

Custom Metrics
--------------

You can also implement custom evaluation metrics:

.. code-block:: python

   from ragas.metrics.base import MetricWithLLM
   
   class DomainSpecificAccuracy(MetricWithLLM):
       """Custom metric for domain-specific accuracy"""
       
       name: str = "domain_specific_accuracy"
       
       def __init__(self):
           super().__init__()
       
       def _compute_score(self, row):
           """Compute domain-specific accuracy score"""
           
           question = row['question']
           answer = row['answer']
           
           # Custom evaluation logic
           domain_terms = self._extract_domain_terms(question)
           answer_coverage = self._check_domain_coverage(answer, domain_terms)
           
           return answer_coverage
       
       def _extract_domain_terms(self, question):
           """Extract domain-specific terms from question"""
           # Implementation specific to your domain
           pass
       
       def _check_domain_coverage(self, answer, terms):
           """Check how well answer covers domain terms"""
           # Implementation specific to your needs
           pass

Automated Monitoring
--------------------

Set up automated metric monitoring:

.. code-block:: python

   import schedule
   import time
   from datetime import datetime

   def automated_evaluation():
       """Run automated evaluation and alert on score drops"""
       
       # Run evaluation
       results = run_ragas_evaluation()
       
       # Calculate current averages
       current_scores = results.mean()
       
       # Load historical baselines
       baselines = load_baseline_scores()
       
       # Check for significant drops
       for metric, current in current_scores.items():
           baseline = baselines.get(metric, 0.7)
           
           if current < baseline * 0.9:  # 10% drop threshold
               send_alert(f"{metric} dropped to {current:.3f} (baseline: {baseline:.3f})")
       
       # Save current results
       save_evaluation_results(results, datetime.now())

   # Schedule daily evaluation
   schedule.every().day.at("02:00").do(automated_evaluation)

Best Practices
--------------

1. **Baseline Establishment**:
   - Run initial evaluation to establish baselines
   - Use consistent test datasets
   - Document evaluation conditions

2. **Regular Monitoring**:
   - Schedule periodic evaluations
   - Track metric trends over time
   - Set up alerting for significant changes

3. **Metric Interpretation**:
   - Consider metrics together, not in isolation
   - Understand metric limitations and biases
   - Use domain expertise for validation

4. **Continuous Improvement**:
   - Use metrics to guide optimization efforts
   - A/B test configuration changes
   - Validate improvements with real users

Next Steps
----------

- **Set Up Monitoring**: :doc:`setup` - Configure automated evaluation
- **RAGAS Deep Dive**: :doc:`ragas` - Comprehensive RAGAS guide
- **Custom Evaluation**: Create domain-specific evaluation metrics