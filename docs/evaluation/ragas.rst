RAGAS Evaluation System
=======================

RAGAS (Retrieval-Augmented Generation Assessment) is a comprehensive framework for evaluating RAG systems. Agentic RAG includes a complete RAGAS evaluation pipeline with Qdrant integration.

Overview
--------

The RAGAS evaluation system in Agentic RAG provides:

- **Automated Evaluation**: End-to-end evaluation pipeline
- **Multiple Metrics**: Comprehensive assessment across different dimensions
- **Qdrant Integration**: Native support for Qdrant vector database
- **Azure OpenAI Support**: Evaluation using Azure OpenAI models
- **CSV Reporting**: Detailed results export for analysis

Key Features
------------

🎯 **Comprehensive Metrics**
   - Context Precision: Quality of retrieved context
   - Context Recall: Completeness of retrieved context
   - Faithfulness: Alignment between answer and context
   - Answer Relevancy: Relevance of answer to question
   - Answer Correctness: Overall accuracy of responses

📊 **Automated Reporting**
   - CSV export with detailed metrics
   - Per-question breakdown
   - Aggregate statistics
   - Performance benchmarking

🔄 **Hybrid Search Evaluation**
   - Tests both semantic and keyword search
   - Evaluates MMR diversification
   - Assesses multi-sector performance

Quick Start
-----------

**1. Basic Evaluation Run**

.. code-block:: bash

   cd test_ragas
   python rag_ragas_qdrant.py

**2. View Results**

Results are saved to `test_ragas/output/ragas_metrics.csv`:

.. code-block:: csv

   question,answer,contexts,ground_truth,context_precision,context_recall,faithfulness,answer_relevancy,answer_correctness
   "What is Basel III?","Basel III is...","[context1, context2]","Basel III regulation...",0.85,0.92,0.78,0.88,0.83

**3. Analyze Results**

.. code-block:: python

   import pandas as pd
   
   # Load results
   df = pd.read_csv('test_ragas/output/ragas_metrics.csv')
   
   # Calculate average scores
   avg_scores = df[['context_precision', 'context_recall', 
                   'faithfulness', 'answer_relevancy', 
                   'answer_correctness']].mean()
   
   print("Average RAGAS Scores:")
   for metric, score in avg_scores.items():
       print(f"{metric}: {score:.3f}")

Evaluation Metrics
------------------

Context Precision
~~~~~~~~~~~~~~~~~

**What it measures**: Quality and relevance of retrieved context chunks.

- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: 
  - 1.0 = All retrieved context is highly relevant
  - 0.5 = Half the context is relevant
  - 0.0 = No relevant context retrieved

**Example**:

.. code-block:: python

   # High precision: All 3 chunks are relevant to "Basel III requirements"
   contexts = [
       "Basel III capital requirements...",
       "Basel III liquidity ratios...", 
       "Basel III implementation timeline..."
   ]
   # Score: ~0.95

   # Low precision: Only 1 of 3 chunks is relevant
   contexts = [
       "Basel III capital requirements...",
       "Weather forecast for tomorrow...",
       "Recipe for chocolate cake..."
   ]
   # Score: ~0.33

Context Recall
~~~~~~~~~~~~~~

**What it measures**: Completeness of retrieved context relative to ground truth.

- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**:
  - 1.0 = All relevant information was retrieved
  - 0.5 = Half of relevant information was retrieved
  - 0.0 = No relevant information was retrieved

**Example**:

.. code-block:: python

   # High recall: Retrieved contexts cover most ground truth aspects
   ground_truth = "Basel III includes capital, liquidity, and leverage requirements"
   contexts = [
       "Basel III capital requirements: 8% minimum...",
       "Basel III liquidity coverage ratio...",
       "Basel III leverage ratio requirements..."
   ]
   # Score: ~0.90

   # Low recall: Retrieved contexts miss key aspects
   contexts = [
       "Basel III capital requirements: 8% minimum..."
   ]
   # Score: ~0.30

Faithfulness
~~~~~~~~~~~~

**What it measures**: How well the generated answer aligns with the retrieved context.

- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**:
  - 1.0 = Answer is completely supported by context
  - 0.5 = Answer is partially supported by context
  - 0.0 = Answer contradicts or ignores context

**Example**:

.. code-block:: python

   # High faithfulness
   context = "Basel III requires banks to maintain 8% capital ratio"
   answer = "According to Basel III, banks must maintain a minimum 8% capital ratio"
   # Score: ~0.95

   # Low faithfulness
   context = "Basel III requires banks to maintain 8% capital ratio"
   answer = "Basel III requires a 15% capital ratio"  # Incorrect information
   # Score: ~0.10

Answer Relevancy
~~~~~~~~~~~~~~~~

**What it measures**: How well the answer addresses the specific question asked.

- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**:
  - 1.0 = Answer directly and completely addresses the question
  - 0.5 = Answer partially addresses the question
  - 0.0 = Answer doesn't address the question

**Example**:

.. code-block:: python

   # High relevancy
   question = "What are the main Basel III capital requirements?"
   answer = "Basel III requires banks to maintain 8% minimum capital ratio..."
   # Score: ~0.90

   # Low relevancy
   question = "What are the main Basel III capital requirements?"
   answer = "Banking has a long history dating back to ancient times..."
   # Score: ~0.15

Answer Correctness
~~~~~~~~~~~~~~~~~~

**What it measures**: Overall accuracy and completeness of the answer compared to ground truth.

- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**:
  - 1.0 = Answer is completely correct and comprehensive
  - 0.5 = Answer is partially correct
  - 0.0 = Answer is incorrect

**Example**:

.. code-block:: python

   # High correctness
   ground_truth = "Basel III requires 8% capital ratio and liquidity coverage"
   answer = "Basel III mandates 8% minimum capital ratio and liquidity coverage requirements"
   # Score: ~0.88

   # Low correctness
   answer = "Basel III requires 5% capital ratio"  # Factually incorrect
   # Score: ~0.25

Evaluation Setup
----------------

Dataset Preparation
~~~~~~~~~~~~~~~~~~~

The evaluation system uses a structured dataset:

.. code-block:: python

   # Example evaluation dataset
   evaluation_dataset = [
       {
           "question": "What are Basel III capital requirements?",
           "ground_truth": "Basel III requires banks to maintain a minimum 8% capital adequacy ratio...",
           "sector": "bancario"
       },
       {
           "question": "How does renewable energy affect grid stability?",
           "ground_truth": "Renewable energy sources introduce variability that affects grid stability...",
           "sector": "energia"
       }
   ]

Custom Test Questions
~~~~~~~~~~~~~~~~~~~~~

You can add custom test questions by editing the dataset in `rag_ragas_qdrant.py`:

.. code-block:: python

   def create_evaluation_dataset():
       """Create evaluation dataset with questions and ground truth"""
       
       return [
           {
               "question": "Your custom question here",
               "ground_truth": "Expected comprehensive answer",
               "sector": "relevant_sector"
           },
           # Add more questions...
       ]

Running Evaluations
-------------------

Basic Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd test_ragas
   python rag_ragas_qdrant.py

This will:

1. Initialize Qdrant connection
2. Index sample documents if needed
3. Run evaluation on predefined dataset
4. Generate and save metrics to CSV

Custom Evaluation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rag_ragas_qdrant import QdrantRagasEvaluator
   
   # Initialize evaluator
   evaluator = QdrantRagasEvaluator()
   
   # Custom evaluation dataset
   custom_dataset = [
       {
           "question": "Your question",
           "ground_truth": "Expected answer",
           "sector": "bancario"
       }
   ]
   
   # Run evaluation
   results = evaluator.evaluate_dataset(custom_dataset)
   
   # Save results
   evaluator.save_results(results, "custom_evaluation.csv")

Sector-Specific Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Evaluate specific sector
   banking_questions = [
       {
           "question": "What is regulatory capital?",
           "ground_truth": "Regulatory capital refers to...",
           "sector": "bancario"
       }
   ]
   
   results = evaluator.evaluate_dataset(
       banking_questions, 
       collection_name="agentic_rag_docs_bancario"
   )

Evaluation Analysis
-------------------

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~

**Good RAGAS Scores** (for most RAG systems):

- Context Precision: > 0.70
- Context Recall: > 0.65
- Faithfulness: > 0.75
- Answer Relevancy: > 0.80
- Answer Correctness: > 0.70

**Excellent RAGAS Scores**:

- Context Precision: > 0.85
- Context Recall: > 0.80
- Faithfulness: > 0.90
- Answer Relevancy: > 0.90
- Answer Correctness: > 0.85

Score Interpretation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def interpret_ragas_scores(df):
       """Interpret RAGAS evaluation results"""
       
       avg_scores = df[['context_precision', 'context_recall', 
                       'faithfulness', 'answer_relevancy', 
                       'answer_correctness']].mean()
       
       interpretations = {}
       
       for metric, score in avg_scores.items():
           if score >= 0.85:
               level = "Excellent"
           elif score >= 0.70:
               level = "Good"
           elif score >= 0.50:
               level = "Fair"
           else:
               level = "Needs Improvement"
           
           interpretations[metric] = {
               "score": score,
               "level": level
           }
       
       return interpretations

Improving Scores
~~~~~~~~~~~~~~~~

**Low Context Precision**:
- Improve document chunking strategy
- Adjust hybrid search weights
- Enhance query preprocessing

**Low Context Recall**:
- Increase search result count (top_k)
- Improve document coverage in index
- Adjust MMR diversity parameters

**Low Faithfulness**:
- Improve prompt engineering
- Use more conservative generation parameters
- Enhance context formatting

**Low Answer Relevancy**:
- Improve question understanding in agents
- Enhance routing logic in Check Crew
- Better context selection algorithms

**Low Answer Correctness**:
- Improve knowledge base quality
- Better ground truth preparation
- Enhanced synthesis agent prompts

Advanced Evaluation
-------------------

Batch Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Large-scale evaluation
   def run_batch_evaluation(questions_file, batch_size=10):
       evaluator = QdrantRagasEvaluator()
       
       # Load questions from file
       with open(questions_file, 'r') as f:
           questions = json.load(f)
       
       # Process in batches
       all_results = []
       for i in range(0, len(questions), batch_size):
           batch = questions[i:i+batch_size]
           results = evaluator.evaluate_dataset(batch)
           all_results.extend(results)
       
       return all_results

Cross-Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   # K-fold cross-validation for robust evaluation
   from sklearn.model_selection import KFold
   
   def cross_validate_rag(dataset, k=5):
       kfold = KFold(n_splits=k, shuffle=True, random_state=42)
       scores = []
       
       for train_idx, test_idx in kfold.split(dataset):
           test_data = [dataset[i] for i in test_idx]
           results = evaluator.evaluate_dataset(test_data)
           
           # Calculate average scores for this fold
           fold_scores = calculate_average_scores(results)
           scores.append(fold_scores)
       
       return scores

A/B Testing
~~~~~~~~~~~

.. code-block:: python

   # Compare different RAG configurations
   def compare_rag_configs(dataset, config_a, config_b):
       # Evaluate with configuration A
       evaluator_a = QdrantRagasEvaluator(config_a)
       results_a = evaluator_a.evaluate_dataset(dataset)
       
       # Evaluate with configuration B
       evaluator_b = QdrantRagasEvaluator(config_b)
       results_b = evaluator_b.evaluate_dataset(dataset)
       
       # Compare results
       comparison = compare_results(results_a, results_b)
       return comparison

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**1. Low Scores Across All Metrics**
   - Check document quality and relevance
   - Verify Qdrant indexing completed successfully
   - Ensure embedding model consistency

**2. Evaluation Takes Too Long**
   - Reduce dataset size for testing
   - Increase batch processing
   - Optimize Qdrant connection settings

**3. Inconsistent Results**
   - Set random seeds for reproducibility
   - Use larger evaluation datasets
   - Check for model temperature settings

**4. Memory Issues**
   - Reduce batch sizes
   - Clear cache between evaluations
   - Monitor system resources

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimized evaluation configuration
   EVAL_CONFIG = {
       "batch_size": 5,           # Smaller batches for memory
       "max_workers": 2,          # Parallel processing
       "timeout": 60,             # Per-question timeout
       "cache_embeddings": True,  # Cache for reuse
       "async_evaluation": True   # Non-blocking evaluation
   }

Next Steps
----------

After setting up RAGAS evaluation:

1. **Understand Metrics**: :doc:`metrics` - Deep dive into each metric
2. **Evaluation Setup**: :doc:`setup` - Advanced configuration options
3. **Custom Evaluation**: Create domain-specific evaluation datasets
4. **Continuous Monitoring**: Set up automated evaluation pipelines