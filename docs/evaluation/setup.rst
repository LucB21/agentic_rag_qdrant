Evaluation Setup
================

This guide covers advanced setup and configuration for RAGAS evaluation in Agentic RAG.

Prerequisites
-------------

Before setting up evaluation:

- ✅ Completed :doc:`../installation`
- ✅ Configured Azure OpenAI credentials
- ✅ Qdrant instance running with indexed documents
- ✅ Python virtual environment activated

Environment Setup
-----------------

**1. Install Evaluation Dependencies**

The evaluation dependencies are included in the main requirements:

.. code-block:: bash

   # Included in main installation
   pip install ragas>=0.3.5
   pip install pandas>=1.5.0
   pip install datasets>=2.0.0

**2. Verify RAGAS Installation**

.. code-block:: python

   import ragas
   from ragas.metrics import (
       context_precision,
       context_recall,
       faithfulness,
       answer_relevancy,
       answer_correctness
   )
   
   print(f"RAGAS version: {ragas.__version__}")
   print("All metrics imported successfully!")

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Add evaluation-specific variables to your `.env` file:

.. code-block:: bash

   # RAGAS Evaluation Configuration
   RAGAS_BATCH_SIZE=5
   RAGAS_MAX_RETRIES=3
   RAGAS_TIMEOUT=60
   RAGAS_ASYNC_MODE=true
   
   # Evaluation Dataset
   EVAL_DATASET_SIZE=20
   EVAL_OUTPUT_DIR=test_ragas/output
   
   # Performance Settings
   EVAL_PARALLEL_REQUESTS=3
   EVAL_RATE_LIMIT=10

Evaluation Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create `test_ragas/config.yaml`:

.. code-block:: yaml

   evaluation:
     metrics:
       - context_precision
       - context_recall
       - faithfulness
       - answer_relevancy
       - answer_correctness
     
     dataset:
       size: 20
       sectors: ["bancario", "energia"]
       question_types: ["factual", "analytical", "comparative"]
     
     models:
       llm:
         provider: "azure_openai"
         model: "gpt-4o"
         temperature: 0.1
         max_tokens: 4000
       
       embeddings:
         provider: "azure_openai"
         model: "text-embedding-ada-002"
         dimensions: 1536
     
     qdrant:
       collection_prefix: "agentic_rag_docs"
       search_params:
         top_k: 10
         score_threshold: 0.3
     
     output:
       format: "csv"
       include_raw_data: true
       save_intermediate: false

Loading Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import yaml
   from pathlib import Path

   def load_evaluation_config():
       """Load evaluation configuration from YAML file"""
       
       config_path = Path("test_ragas/config.yaml")
       
       if config_path.exists():
           with open(config_path, 'r') as f:
               config = yaml.safe_load(f)
           return config
       else:
           # Return default configuration
           return {
               "evaluation": {
                   "metrics": ["context_precision", "context_recall", 
                              "faithfulness", "answer_relevancy", "answer_correctness"],
                   "dataset": {"size": 10},
                   "output": {"format": "csv"}
               }
           }

Dataset Preparation
-------------------

Ground Truth Dataset
~~~~~~~~~~~~~~~~~~~~

Create a comprehensive ground truth dataset:

.. code-block:: python

   # test_ragas/ground_truth_dataset.py
   
   GROUND_TRUTH_DATASET = [
       {
           "question": "What are the main components of Basel III?",
           "ground_truth": """Basel III consists of three main components:
           1. Capital requirements: Minimum 8% capital adequacy ratio with additional buffers
           2. Liquidity requirements: Liquidity Coverage Ratio (LCR) and Net Stable Funding Ratio (NSFR)
           3. Leverage ratio: Minimum 3% leverage ratio to prevent excessive leverage""",
           "sector": "bancario",
           "difficulty": "medium",
           "question_type": "factual"
       },
       {
           "question": "How do smart grids improve energy efficiency?",
           "ground_truth": """Smart grids improve energy efficiency through:
           1. Real-time monitoring and control of energy flow
           2. Demand response programs that adjust consumption during peak times
           3. Integration of renewable energy sources with better forecasting
           4. Reduced transmission losses through optimized routing""",
           "sector": "energia",
           "difficulty": "medium", 
           "question_type": "analytical"
       },
       {
           "question": "Compare risk management in banking vs energy sector",
           "ground_truth": """Risk management differs between sectors:
           Banking: Focus on credit risk, market risk, operational risk, regulatory compliance
           Energy: Emphasis on commodity price risk, weather risk, regulatory changes, infrastructure risk
           Both sectors share: Operational risk, cybersecurity risk, ESG risk management""",
           "sector": ["bancario", "energia"],
           "difficulty": "hard",
           "question_type": "comparative"
       }
   ]

Question Generation
~~~~~~~~~~~~~~~~~~~

Automatically generate test questions:

.. code-block:: python

   from azure_openai import AzureOpenAI
   import json

   def generate_test_questions(documents, sector, count=10):
       """Generate test questions from documents using LLM"""
       
       client = AzureOpenAI()
       
       # Sample document content
       doc_sample = "\n".join([doc.page_content[:500] for doc in documents[:5]])
       
       prompt = f"""
       Based on the following {sector} sector documents, generate {count} diverse test questions 
       that would be good for evaluating a RAG system.
       
       Include a mix of:
       - Factual questions (asking for specific information)
       - Analytical questions (requiring reasoning)
       - Comparative questions (comparing concepts)
       
       Document content:
       {doc_sample}
       
       Return as JSON array with format:
       [
           {{
               "question": "Question text",
               "expected_answer_type": "factual|analytical|comparative",
               "difficulty": "easy|medium|hard"
           }}
       ]
       """
       
       response = client.chat.completions.create(
           model="gpt-4o",
           messages=[{"role": "user", "content": prompt}],
           temperature=0.7
       )
       
       try:
           questions = json.loads(response.choices[0].message.content)
           return questions
       except:
           return []

Answer Generation
~~~~~~~~~~~~~~~~~

Generate ground truth answers:

.. code-block:: python

   def generate_ground_truth_answers(questions, documents):
       """Generate comprehensive ground truth answers"""
       
       client = AzureOpenAI()
       ground_truth_data = []
       
       for question_data in questions:
           question = question_data["question"]
           
           # Find most relevant documents
           relevant_docs = search_relevant_docs(question, documents, top_k=5)
           context = "\n".join([doc.page_content for doc in relevant_docs])
           
           prompt = f"""
           Based on the provided context, write a comprehensive and accurate answer 
           to the question. The answer will be used as ground truth for evaluation.
           
           Question: {question}
           
           Context: {context}
           
           Requirements:
           - Be comprehensive and accurate
           - Include specific details and numbers when available
           - Structure the answer clearly
           - Only use information from the provided context
           """
           
           response = client.chat.completions.create(
               model="gpt-4o",
               messages=[{"role": "user", "content": prompt}],
               temperature=0.1
           )
           
           ground_truth_data.append({
               "question": question,
               "ground_truth": response.choices[0].message.content,
               "sector": question_data.get("sector", "general"),
               "difficulty": question_data.get("difficulty", "medium"),
               "question_type": question_data.get("expected_answer_type", "factual")
           })
       
       return ground_truth_data

Evaluation Pipeline Setup
-------------------------

Basic Evaluation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # test_ragas/evaluation_pipeline.py
   
   import asyncio
   from pathlib import Path
   import pandas as pd
   from datetime import datetime
   
   class EvaluationPipeline:
       """Complete evaluation pipeline for RAGAS"""
       
       def __init__(self, config_path="config.yaml"):
           self.config = self.load_config(config_path)
           self.results_dir = Path(self.config["evaluation"]["output"].get("dir", "output"))
           self.results_dir.mkdir(exist_ok=True)
       
       def load_config(self, config_path):
           """Load configuration from file"""
           with open(config_path, 'r') as f:
               return yaml.safe_load(f)
       
       async def run_evaluation(self, dataset=None):
           """Run complete evaluation pipeline"""
           
           if dataset is None:
               dataset = self.load_default_dataset()
           
           print(f"Starting evaluation with {len(dataset)} questions...")
           
           # Initialize evaluator
           evaluator = QdrantRagasEvaluator(self.config)
           
           # Run evaluation in batches
           batch_size = self.config["evaluation"].get("batch_size", 5)
           all_results = []
           
           for i in range(0, len(dataset), batch_size):
               batch = dataset[i:i+batch_size]
               print(f"Processing batch {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1}")
               
               try:
                   batch_results = await evaluator.evaluate_batch_async(batch)
                   all_results.extend(batch_results)
               except Exception as e:
                   print(f"Error in batch {i//batch_size + 1}: {e}")
                   continue
           
           # Save results
           self.save_results(all_results)
           
           # Generate report
           self.generate_report(all_results)
           
           return all_results
       
       def save_results(self, results):
           """Save evaluation results"""
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           
           # Save detailed results
           df = pd.DataFrame(results)
           csv_path = self.results_dir / f"ragas_evaluation_{timestamp}.csv"
           df.to_csv(csv_path, index=False)
           
           # Save summary statistics
           summary = self.calculate_summary_stats(df)
           summary_path = self.results_dir / f"evaluation_summary_{timestamp}.json"
           
           with open(summary_path, 'w') as f:
               json.dump(summary, f, indent=2)
           
           print(f"Results saved to {csv_path}")
           print(f"Summary saved to {summary_path}")
       
       def calculate_summary_stats(self, df):
           """Calculate summary statistics"""
           metrics = ['context_precision', 'context_recall', 'faithfulness', 
                     'answer_relevancy', 'answer_correctness']
           
           summary = {
               "overall": {
                   metric: {
                       "mean": float(df[metric].mean()),
                       "std": float(df[metric].std()),
                       "min": float(df[metric].min()),
                       "max": float(df[metric].max())
                   } for metric in metrics if metric in df.columns
               }
           }
           
           # Sector-specific statistics
           if 'sector' in df.columns:
               summary["by_sector"] = {}
               for sector in df['sector'].unique():
                   sector_df = df[df['sector'] == sector]
                   summary["by_sector"][sector] = {
                       metric: float(sector_df[metric].mean())
                       for metric in metrics if metric in sector_df.columns
                   }
           
           return summary

Advanced Configuration
----------------------

Custom Metrics Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings

   class CustomEvaluationSuite:
       """Custom evaluation suite with domain-specific metrics"""
       
       def __init__(self):
           self.standard_metrics = [
               context_precision,
               context_recall,
               faithfulness,
               answer_relevancy,
               answer_correctness
           ]
           
           self.custom_metrics = [
               DomainSpecificAccuracy(),
               TechnicalTermUsage(),
               RegulationCompliance()
           ]
       
       def evaluate(self, dataset):
           """Run evaluation with both standard and custom metrics"""
           
           # Standard RAGAS evaluation
           standard_results = evaluate(
               dataset=dataset,
               metrics=self.standard_metrics
           )
           
           # Custom metric evaluation
           custom_results = self.evaluate_custom_metrics(dataset)
           
           # Combine results
           combined_results = self.combine_results(standard_results, custom_results)
           
           return combined_results

Parallel Evaluation Setup
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from concurrent.futures import ThreadPoolExecutor
   
   class ParallelEvaluator:
       """Parallel evaluation for faster processing"""
       
       def __init__(self, max_workers=3):
           self.max_workers = max_workers
           self.executor = ThreadPoolExecutor(max_workers=max_workers)
       
       async def evaluate_parallel(self, dataset, batch_size=5):
           """Evaluate dataset in parallel batches"""
           
           # Split dataset into batches
           batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
           
           # Create evaluation tasks
           tasks = []
           for batch in batches:
               task = asyncio.create_task(self.evaluate_batch_async(batch))
               tasks.append(task)
           
           # Wait for all tasks to complete
           batch_results = await asyncio.gather(*tasks, return_exceptions=True)
           
           # Combine results
           all_results = []
           for result in batch_results:
               if isinstance(result, Exception):
                   print(f"Batch failed: {result}")
                   continue
               all_results.extend(result)
           
           return all_results

Monitoring and Alerts
---------------------

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   from dataclasses import dataclass
   from typing import List, Dict

   @dataclass
   class EvaluationMetrics:
       """Evaluation performance metrics"""
       total_questions: int
       successful_evaluations: int
       failed_evaluations: int
       average_time_per_question: float
       total_time: float
       error_rate: float

   class EvaluationMonitor:
       """Monitor evaluation performance and health"""
       
       def __init__(self):
           self.start_time = None
           self.question_times = []
           self.errors = []
       
       def start_monitoring(self):
           """Start performance monitoring"""
           self.start_time = time.time()
           self.question_times = []
           self.errors = []
       
       def log_question_completion(self, duration: float):
           """Log completion of a question evaluation"""
           self.question_times.append(duration)
       
       def log_error(self, error: Exception):
           """Log evaluation error"""
           self.errors.append(str(error))
       
       def get_metrics(self) -> EvaluationMetrics:
           """Get current performance metrics"""
           total_time = time.time() - self.start_time if self.start_time else 0
           total_questions = len(self.question_times) + len(self.errors)
           
           return EvaluationMetrics(
               total_questions=total_questions,
               successful_evaluations=len(self.question_times),
               failed_evaluations=len(self.errors),
               average_time_per_question=sum(self.question_times) / len(self.question_times) if self.question_times else 0,
               total_time=total_time,
               error_rate=len(self.errors) / total_questions if total_questions > 0 else 0
           )

Automated Quality Checks
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class QualityChecker:
       """Automated quality checks for evaluation results"""
       
       def __init__(self, thresholds=None):
           self.thresholds = thresholds or {
               "context_precision": 0.6,
               "context_recall": 0.6,
               "faithfulness": 0.7,
               "answer_relevancy": 0.7,
               "answer_correctness": 0.6
           }
       
       def check_quality(self, results_df):
           """Run quality checks on evaluation results"""
           
           issues = []
           
           for metric, threshold in self.thresholds.items():
               if metric in results_df.columns:
                   avg_score = results_df[metric].mean()
                   if avg_score < threshold:
                       issues.append(f"{metric} below threshold: {avg_score:.3f} < {threshold}")
           
           # Check for outliers
           outliers = self.detect_outliers(results_df)
           if outliers:
               issues.extend(outliers)
           
           # Check for missing data
           missing_data = self.check_missing_data(results_df)
           if missing_data:
               issues.extend(missing_data)
           
           return issues
       
       def detect_outliers(self, df):
           """Detect outlier scores that may indicate issues"""
           outliers = []
           
           for metric in self.thresholds.keys():
               if metric in df.columns:
                   Q1 = df[metric].quantile(0.25)
                   Q3 = df[metric].quantile(0.75)
                   IQR = Q3 - Q1
                   
                   lower_bound = Q1 - 1.5 * IQR
                   upper_bound = Q3 + 1.5 * IQR
                   
                   outlier_count = ((df[metric] < lower_bound) | (df[metric] > upper_bound)).sum()
                   
                   if outlier_count > len(df) * 0.1:  # More than 10% outliers
                       outliers.append(f"{metric} has {outlier_count} outliers ({outlier_count/len(df)*100:.1f}%)")
           
           return outliers

Continuous Integration
----------------------

CI/CD Pipeline Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

Create `.github/workflows/evaluation.yml`:

.. code-block:: yaml

   name: RAGAS Evaluation
   
   on:
     schedule:
       - cron: '0 2 * * *'  # Daily at 2 AM
     workflow_dispatch:
   
   jobs:
     evaluate:
       runs-on: ubuntu-latest
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python
         uses: actions/setup-python@v3
         with:
           python-version: '3.10'
       
       - name: Install dependencies
         run: |
           pip install -r requirements.txt
       
       - name: Run RAGAS evaluation
         env:
           AZURE_OPENAI_API_KEY: ${{ secrets.AZURE_OPENAI_API_KEY }}
           AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
           QDRANT_URL: ${{ secrets.QDRANT_URL }}
           QDRANT_API_KEY: ${{ secrets.QDRANT_API_KEY }}
         run: |
           cd test_ragas
           python evaluation_pipeline.py
       
       - name: Upload results
         uses: actions/upload-artifact@v3
         with:
           name: evaluation-results
           path: test_ragas/output/

Next Steps
----------

After setting up evaluation:

1. **Run First Evaluation**: Execute the pipeline to establish baselines
2. **Review Metrics**: :doc:`metrics` - Understand evaluation results
3. **Optimize System**: Use results to improve RAG performance
4. **Automate Monitoring**: Set up continuous evaluation and alerting