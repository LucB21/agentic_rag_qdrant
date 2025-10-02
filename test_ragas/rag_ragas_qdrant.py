from __future__ import annotations

"""
RAG evaluation with RAGAS framework using Qdrant as vector database.

This module combines:
- Qdrant vector database for high-performance semantic search
- Azure OpenAI for embeddings and chat
- RAGAS framework for comprehensive RAG evaluation
- Real document loading (PDF, TXT, MD, DOCX)

Usage:
    python rag_ragas_qdrant.py

Dependencies:
    pip install qdrant-client langchain langchain-openai langchain-community ragas pandas python-dotenv
"""

import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Dict, Optional, Any, Tuple

from dotenv import load_dotenv

# LangChain core
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Azure OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
try:
    from langchain_community.document_loaders import Docx2txtLoader
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    PayloadSchemaType,
    FieldCondition,
    MatchValue,
    MatchText,
    Filter,
    SearchParams,
    PointStruct,
)

# RAGAS
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
)
from ragas.metrics import AnswerRelevancy
import pandas as pd

load_dotenv()

@dataclass
class Settings:
    # =========================
    # Document Path Configuration
    # =========================
    PATH_TO_DOCUMENTS: str = "PATH_TO_DOCUMENTS"
    """Environment variable name for the path to documents directory."""
    
    # =========================
    # Qdrant Configuration
    # =========================
    qdrant_url: str = "http://localhost:6333"
    """Qdrant server URL."""
    
    collection: str = "rag_chunks"
    """Collection name for storing document chunks and vectors."""
    
    # =========================
    # Document Chunking Configuration
    # =========================
    chunk_size: int = 700
    """Maximum number of characters per document chunk."""
    
    chunk_overlap: int = 120
    """Number of characters to overlap between consecutive chunks."""
    
    # =========================
    # Search Configuration
    # =========================
    top_n_semantic: int = 30
    """Number of top semantic search candidates to retrieve initially."""
    
    top_n_text: int = 100
    """Maximum number of text-based matches to consider for hybrid fusion."""
    
    final_k: int = 6
    """Final number of results to return after all processing steps."""
    
    alpha: float = 0.75
    """Weight for semantic similarity in hybrid score fusion (0.0 to 1.0)."""
    
    text_boost: float = 0.20
    """Additional score boost for results that match both semantic and text criteria."""
    
    # =========================
    # MMR Configuration
    # =========================
    use_mmr: bool = True
    """Whether to use MMR for result diversification."""
    
    mmr_lambda: float = 0.6
    """MMR diversification parameter balancing relevance vs. diversity (0.0 to 1.0)."""
    
    # =========================
    # Azure OpenAI Configuration
    # =========================
    azure_api_key_env: str = "AZURE_OPENAI_API_KEY"
    azure_api_version_env: str = "AZURE_OPENAI_API_VERSION"
    azure_endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    azure_chat_deployment_env: str = "AZURE_OPENAI_CHAT_DEPLOYMENT"
    azure_embed_deployment_env: str = "AZURE_OPENAI_EMBED_DEPLOYMENT"
    
    temperature: float = 0.0
    """Temperature setting for LLM generation."""
    
    max_tokens: int = 800
    """Maximum tokens for LLM response."""

SETTINGS = Settings()

# =========================
# Azure OpenAI Components
# =========================

def get_azure_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    """Create Azure OpenAI embeddings client."""
    endpoint = os.getenv(settings.azure_endpoint_env)
    api_key = os.getenv(settings.azure_api_key_env)
    api_version = os.getenv(settings.azure_api_version_env)
    embed_deployment = os.getenv(settings.azure_embed_deployment_env)

    if not (endpoint and api_key and api_version and embed_deployment):
        raise RuntimeError(
            "Config Azure OpenAI mancante: assicurati di impostare ENDPOINT/API_KEY/API_VERSION e il deployment embeddings."
        )

    return AzureOpenAIEmbeddings(
        azure_deployment=embed_deployment,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
    )

def get_azure_chat_llm(settings: Settings) -> AzureChatOpenAI:
    """Create Azure OpenAI chat LLM client."""
    endpoint = os.getenv(settings.azure_endpoint_env)
    api_key = os.getenv(settings.azure_api_key_env)
    api_version = os.getenv(settings.azure_api_version_env)
    chat_deployment = os.getenv(settings.azure_chat_deployment_env)
    
    if not (endpoint and api_key and api_version and chat_deployment):
        raise RuntimeError(
            "Config Azure OpenAI mancante: assicurati di impostare ENDPOINT/API_KEY/API_VERSION e il deployment chat."
        )
    
    # Test the LLM connection
    llm = AzureChatOpenAI(
        azure_deployment=chat_deployment,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    
    # Simple test to verify the LLM works
    try:
        test_response = llm.invoke("test")
        if test_response:
            print("Azure LLM configured successfully")
            return llm
        else:
            print("Azure LLM test failed")
            return None
    except Exception as e:
        print(f"Azure LLM configuration error: {e}")
        return None

def get_azure_embedding_dimension(embeddings: AzureOpenAIEmbeddings) -> int:
    """Get the dimension of Azure OpenAI embeddings."""
    try:
        sample_embedding = embeddings.embed_query("test")
        return len(sample_embedding)
    except Exception as e:
        print(f"Failed to get embedding dimension: {e}")
        return 1536  # text-embedding-ada-002 default

# =========================
# Document Loading
# =========================

def load_documents_from_dir(path: str | Path) -> List[Document]:
    """Load supported documents from a directory recursively."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cartella non trovata: {path}")

    docs: List[Document] = []

    def _load_one(p: Path) -> Iterable[Document]:
        suffix = p.suffix.lower()
        if suffix == ".pdf":
            return PyPDFLoader(str(p)).load()
        if suffix in {".txt", ".md"}:
            return TextLoader(str(p), encoding="utf-8").load()
        if suffix == ".docx" and HAS_DOCX:
            return Docx2txtLoader(str(p)).load()
        return []

    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}:
            docs.extend(list(_load_one(p)))

    if not docs:
        raise RuntimeError("Nessun documento supportato trovato nella cartella.")

    # Add readable source for citations
    for i, d in enumerate(docs, start=1):
        d.metadata.setdefault("source", d.metadata.get("source", str(d.metadata.get("file_path", d.metadata.get("source", f"doc_{i}")))))
    return docs

def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["# ", "## ", "### ", "\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
    )
    return splitter.split_documents(docs)

# =========================
# Qdrant Operations
# =========================

def get_qdrant_client(settings: Settings) -> QdrantClient:
    """Get Qdrant client."""
    return QdrantClient(url=settings.qdrant_url)

def recreate_collection_for_rag(client: QdrantClient, settings: Settings, vector_size: int):
    """Create or recreate a Qdrant collection optimized for RAG."""
    client.recreate_collection(
        collection_name=settings.collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(
            m=32,
            ef_construct=256
        ),
        optimizers_config=OptimizersConfigDiff(
            default_segment_number=2
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type="int8", always_ram=False)
        ),
    )

    # Create text index for full-text search
    client.create_payload_index(
        collection_name=settings.collection,
        field_name="text",
        field_schema=PayloadSchemaType.TEXT
    )

    # Create keyword indices
    for key in ["doc_id", "source", "title", "lang"]:
        client.create_payload_index(
            collection_name=settings.collection,
            field_name=key,
            field_schema=PayloadSchemaType.KEYWORD
        )

def build_points(chunks: List[Document], embeds: List[List[float]]) -> List[PointStruct]:
    """Build Qdrant points from chunks and embeddings."""
    pts: List[PointStruct] = []
    for i, (doc, vec) in enumerate(zip(chunks, embeds), start=1):
        payload = {
            "doc_id": doc.metadata.get("id"),
            "source": doc.metadata.get("source"),
            "title": doc.metadata.get("title"),
            "lang": doc.metadata.get("lang", "en"),
            "text": doc.page_content,
            "chunk_id": i - 1
        }
        pts.append(PointStruct(id=i, vector=vec, payload=payload))
    return pts

def upsert_chunks_azure(client: QdrantClient, settings: Settings, chunks: List[Document], embeddings: AzureOpenAIEmbeddings):
    """Upsert document chunks using Azure OpenAI embeddings."""
    vecs = embeddings.embed_documents([c.page_content for c in chunks])
    points = build_points(chunks, vecs)
    client.upsert(collection_name=settings.collection, points=points, wait=True)

# =========================
# Search Operations
# =========================

def qdrant_semantic_search_azure(
    client: QdrantClient,
    settings: Settings,
    query: str,
    embeddings: AzureOpenAIEmbeddings,
    limit: int,
    with_vectors: bool = False
):
    """Perform semantic search using Azure OpenAI embeddings."""
    qv = embeddings.embed_query(query)
    res = client.query_points(
        collection_name=settings.collection,
        query=qv,
        limit=limit,
        with_payload=True,
        with_vectors=with_vectors,
        search_params=SearchParams(
            hnsw_ef=256,
            exact=False
        ),
    )
    return res.points

def qdrant_text_prefilter_ids(
    client: QdrantClient,
    settings: Settings,
    query: str,
    max_hits: int
) -> List[int]:
    """Use full-text index to prefilter points containing keywords."""
    matched_ids: List[int] = []
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=settings.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="text", match=MatchText(text=query))]
            ),
            limit=min(256, max_hits - len(matched_ids)),
            offset=next_page,
            with_payload=False,
            with_vectors=False,
        )
        matched_ids.extend([p.id for p in points])
        if not next_page or len(matched_ids) >= max_hits:
            break
    return matched_ids

def mmr_select(
    query_vec: List[float],
    candidates_vecs: List[List[float]],
    k: int,
    lambda_mult: float
) -> List[int]:
    """Select diverse results using Maximal Marginal Relevance (MMR) algorithm."""
    if not candidates_vecs:
        return []
    V = np.array(candidates_vecs, dtype=float)
    q = np.array(query_vec, dtype=float)

    def cos(a, b):
        na = (a @ a) ** 0.5 + 1e-12
        nb = (b @ b) ** 0.5 + 1e-12
        return float((a @ b) / (na * nb))

    sims = [cos(v, q) for v in V]
    selected: List[int] = []
    remaining = set(range(len(V)))

    while len(selected) < min(k, len(V)):
        if not selected:
            best_idx = max(remaining, key=lambda i: sims[i])
        else:
            best_idx = None
            best_score = -1e9
            for i in remaining:
                rel = lambda_mult * sims[i]
                div = (1 - lambda_mult) * max(cos(V[i], V[j]) for j in selected)
                score = rel - div
                if score > best_score:
                    best_score = score
                    best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected

def hybrid_search_azure(
    client: QdrantClient,
    settings: Settings,
    query: str,
    embeddings: AzureOpenAIEmbeddings
):
    """Perform hybrid search using Azure OpenAI embeddings."""
    # Semantic search
    sem = qdrant_semantic_search_azure(
        client, settings, query, embeddings,
        limit=settings.top_n_semantic, with_vectors=True
    )
    if not sem:
        return []

    # Text-based prefiltering
    text_ids = set(qdrant_text_prefilter_ids(client, settings, query, settings.top_n_text))

    # Score normalization
    scores = [p.score for p in sem]
    smin, smax = min(scores), max(scores)
    def norm(x):
        return 1.0 if smax == smin else (x - smin) / (smax - smin)

    # Score fusion with text boost
    fused: List[Tuple[int, float, Any]] = []
    for idx, p in enumerate(sem):
        base = norm(p.score)
        fuse = settings.alpha * base
        if p.id in text_ids:
            fuse += settings.text_boost
        fused.append((idx, fuse, p))

    # Sort by fused score
    fused.sort(key=lambda t: t[1], reverse=True)

    # Optional MMR for diversification
    if settings.use_mmr:
        qv = embeddings.embed_query(query)
        N = min(len(fused), max(settings.final_k * 5, settings.final_k))
        cut = fused[:N]
        vecs = [sem[i].vector for i, _, _ in cut]
        mmr_idx = mmr_select(qv, vecs, settings.final_k, settings.mmr_lambda)
        picked = [cut[i][2] for i in mmr_idx]
        return picked

    # Return top final_k after fusion
    return [p for _, _, p in fused[:settings.final_k]]

# =========================
# RAG Chain and Retriever
# =========================

def format_docs_for_prompt(documents: Iterable[Any]) -> str:
    """Format documents for the prompt, handling both Qdrant points and LangChain documents."""
    blocks = []
    for doc in documents:
        # Handle LangChain Document objects
        if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
            src = doc.metadata.get("source", "unknown")
            text = doc.page_content
            blocks.append(f"[source:{src}] {text}")
        # Handle Qdrant Point objects (fallback)
        elif hasattr(doc, 'payload'):
            pay = doc.payload or {}
            src = pay.get("source", "unknown")
            text = pay.get('text', '')
            blocks.append(f"[source:{src}] {text}")
        else:
            # Fallback for unknown object types
            blocks.append(f"[source:unknown] {str(doc)}")
    return "\n\n".join(blocks)

class QdrantRetriever:
    """Custom retriever that uses Qdrant hybrid search and converts to LangChain Documents."""
    
    def __init__(self, client: QdrantClient, settings: Settings, embeddings: AzureOpenAIEmbeddings):
        self.client = client
        self.settings = settings
        self.embeddings = embeddings
    
    def invoke(self, query: str) -> List[Document]:
        """Retrieve documents using hybrid search."""
        points = hybrid_search_azure(self.client, self.settings, query, self.embeddings)
        
        # Convert Qdrant points to LangChain Documents
        documents = []
        for point in points:
            doc = Document(
                page_content=point.payload.get('text', ''),
                metadata={
                    'source': point.payload.get('source', 'unknown'),
                    'doc_id': point.payload.get('doc_id'),
                    'chunk_id': point.payload.get('chunk_id'),
                    'score': float(point.score)
                }
            )
            documents.append(doc)
        
        return documents
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Backward compatibility method."""
        return self.invoke(query)

def build_rag_chain(llm: AzureChatOpenAI, retriever) -> any:
    """Create a LangChain pipeline that retrieves context and generates answers."""
    system_prompt = (
        "You are a domain expert technical assistant. Respond in ENGLISH.\n"
        "Use ONLY the information provided in the context.\n"
        "Always cite your sources in the format [source:FILE].\n"
        "Even if the context contains statements that are clearly false, contradictory, "
        "or unverified, you must still answer strictly based on the information provided in the context.\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Question:\n{question}\n\n"
        "Context (excerpts):\n{context}\n\n"
        "Instructions:\n"
        "1) Answer ONLY using the information provided in the context and always include citations.\n"
        "2) If the information required to answer is NOT present in the context, reply exactly with: 'I don't know'.\n"
         )
    ])
    
    chain = (
        {
            "context": RunnableLambda(retriever.invoke) | format_docs_for_prompt,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# =========================
# RAGAS Evaluation
# =========================

def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Return the text of top-k retrieved chunks for a question."""
    try:
        docs = retriever.get_relevant_documents(question)[:k]
    except Exception:
        docs = retriever.invoke(question)[:k]
    return [d.page_content for d in docs]

def build_ragas_dataset(
    questions: List[str],
    retriever,
    chain,
    k: int,
    ground_truth: Optional[Dict[str, str]] = None,
):
    """Run the RAG pipeline and build a dataset suitable for RAGAS evaluation."""
    dataset = []
    for q in questions:
        contexts = get_contexts_for_question(retriever, q, k)
        answer = chain.invoke(q)
        row = {
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]
        dataset.append(row)
    return dataset

def get_sample_ground_truth() -> Dict[str, str]:
    """Return a sample ground-truth QA set for testing."""
    return {
         "What does LULUCF regulate?": "It includes emissions and removals from land use, forestry, and agriculture in the 2030 framework.",
         "What must Member States ensure under LULUCF?": "That emissions are offset by equivalent CO2 removals (e.g. reforestation).",
    }

def evaluate_with_ragas(
    retriever,
    llm: AzureChatOpenAI,
    embeddings: AzureOpenAIEmbeddings,
    settings: Settings,
    questions: List[str],
    ground_truth: Optional[Dict[str, str]] = None,
):
    """Evaluate the pipeline with RAGAS and save metrics to CSV."""
    print("DEBUG: Building RAG chain...")
    chain = build_rag_chain(llm, retriever)
    
    print(f"DEBUG: Building RAGAS dataset with {len(questions)} questions...")
    dataset = build_ragas_dataset(
        questions=questions,
        retriever=retriever,
        chain=chain,
        k=settings.final_k,
        ground_truth=ground_truth,
    )
    
    print("DEBUG: Creating EvaluationDataset...")
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    
    print("DEBUG: Setting up RAGAS metrics...")
    ar = AnswerRelevancy(strictness=1)
    metrics = [context_precision, context_recall, faithfulness, ar]
    if ground_truth:
        metrics.append(answer_correctness)

    print(f"DEBUG: Starting RAGAS evaluation with {len(metrics)} metrics...")
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    print("DEBUG: Processing RAGAS results...")
    df = ragas_result.to_pandas()

    # Rename columns for consistency
    rename_map = {
        "user_input": "question",
        "response": "answer",
        "reference": "ground_truth",
    }
    df_renamed = df.rename(columns=rename_map)

    # Ensure all expected columns exist
    requested_cols = [
        "question",
        "answer",
        "ground_truth",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    ]
    for col in requested_cols:
        if col not in df_renamed.columns:
            df_renamed[col] = pd.NA

    df_out = df_renamed[requested_cols]

    # Save CSV
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ragas_metrics.csv"
    df_out.to_csv(csv_path, sep=';', index=False)

    print(f"\n=== RAGAS metrics (k={settings.final_k}) ===")
    print(df_out.round(4).to_string(index=False))
    print(f"Saved RAGAS CSV to: {csv_path}")
    return df_out

# =========================
# Main Function
# =========================

def rag_tool(question: str) -> Dict[str, Any]:
    """
    Main RAG evaluation function using Qdrant and RAGAS.
    
    This function:
    1. Loads documents and creates Qdrant collection
    2. Performs hybrid search to retrieve relevant chunks
    3. Runs RAGAS evaluation on a sample question set
    4. Returns structured results
    """
    settings = SETTINGS
    
    # Initialize Azure components
    print("Initializing Azure OpenAI components...")
    try:
        embeddings = get_azure_embeddings(settings)
        print("Azure embeddings initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Azure embeddings: {e}")
        return {"error": "Failed to initialize embeddings"}
    
    try:
        llm = get_azure_chat_llm(settings)
        if not llm:
            return {"error": "Failed to initialize LLM"}
    except Exception as e:
        print(f"Failed to initialize Azure LLM: {e}")
        return {"error": "Failed to initialize LLM"}

    # Initialize Qdrant
    print("Initializing Qdrant...")
    client = get_qdrant_client(settings)
    
    # Load and process documents
    print("Loading documents...")
    try:
        docs_path = os.getenv(settings.PATH_TO_DOCUMENTS, "./document")
        docs = load_documents_from_dir(docs_path)
        print(f"Loaded {len(docs)} documents")
    except Exception as e:
        print(f"Failed to load documents: {e}")
        return {"error": "Failed to load documents"}
    
    # Split documents
    print("Splitting documents into chunks...")
    chunks = split_documents(docs, settings)
    print(f"Created {len(chunks)} chunks")
    
    # Create collection
    print("Creating Qdrant collection...")
    vector_size = get_azure_embedding_dimension(embeddings)
    print(f"Using vector size: {vector_size}")
    recreate_collection_for_rag(client, settings, vector_size)
    
    # Upsert chunks
    print("Indexing chunks in Qdrant...")
    upsert_chunks_azure(client, settings, chunks, embeddings)
    print(f"Indexed {len(chunks)} chunks in collection '{settings.collection}'")
    
    # Create retriever
    retriever = QdrantRetriever(client, settings, embeddings)
    
    # Test retrieval with the provided question
    print(f"\nTesting retrieval with question: {question}")
    retrieved_docs = retriever.invoke(question)
    
    print(f"Retrieved {len(retrieved_docs)} documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"  {i}. Source: {doc.metadata.get('source', 'unknown')}")
        print(f"     Score: {doc.metadata.get('score', 'N/A')}")
        print(f"     Preview: {doc.page_content[:200]}...")
        print()
    
    # RAGAS evaluation
    print("Starting RAGAS evaluation...")
    try:
        # Use sample ground truth for evaluation
        gt = get_sample_ground_truth()
        eval_questions = list(gt.keys())
        
        # Add the provided question if it's not in ground truth
        if question not in eval_questions:
            eval_questions.append(question)
        
        print(f"Evaluating {len(eval_questions)} questions...")
        
        evaluate_with_ragas(
            retriever=retriever,
            llm=llm,
            embeddings=embeddings,
            settings=settings,
            questions=eval_questions,
            ground_truth=gt,
        )
        
        print("RAGAS evaluation completed successfully!")
        
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Return structured results
    return {
        "question": question,
        "retrieved_chunks": len(retrieved_docs),
        "chunks_preview": [
            {
                "source": doc.metadata.get('source', 'unknown'),
                "score": doc.metadata.get('score', 'N/A'),
                "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            for doc in retrieved_docs
        ],
        "total_documents": len(docs),
        "total_chunks": len(chunks),
        "collection": settings.collection
    }

if __name__ == "__main__":
    result = rag_tool("What are the main topics covered in the documents?")
    print("\n=== Final Results ===")
    print(f"Question: {result.get('question', 'N/A')}")
    print(f"Retrieved chunks: {result.get('retrieved_chunks', 'N/A')}")
    print(f"Total documents: {result.get('total_documents', 'N/A')}")
    print(f"Total chunks: {result.get('total_chunks', 'N/A')}")