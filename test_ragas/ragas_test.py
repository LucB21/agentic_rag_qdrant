from __future__ import annotations

"""
RAG adattato per Azure OpenAI (chat + embeddings),
con caricamento di documenti REALI (pdf/txt/md/docx) con split robusto,
+ test per verificare e migliorare il prompt
+ **valutazione automatica con RAGAS**.

Esecuzione:
  python rag_azure_real_docs.py --docs ./cartella_documenti \
    --persist ./faiss_index_azure \
    --search-type mmr --k 5 --fetch-k 30 --lambda 0.3

Test "anti-confusione":
  python rag_azure_real_docs.py --run-tests

Dipendenze aggiuntive per la valutazione:
  pip install ragas pandas
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Dict, Optional

from dotenv import load_dotenv
from typing import Type
 
# from crewai.tools import BaseTool
# from pydantic import BaseModel, Field
# from crewai.tools import tool

# LangChain core
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Azure OpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

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

# --- RAGAS ---
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,   # precision@k sui chunk recuperati
    context_recall,      # copertura dei chunk rilevanti
    faithfulness,        # ancoraggio della risposta al contesto
    answer_correctness,  # usa questa solo se hai ground_truth
)
from ragas.metrics import AnswerRelevancy
import pandas as pd

load_dotenv()

@dataclass
class Settings:
    # Persistenza
    persist_dir: str = "faiss_index_azure"
    # Splitting
    chunk_size: int = 700
    chunk_overlap: int = 150
    # Retrieval
    search_type: str = "mmr"  # "mmr" or "similarity"
    k: int = 4
    fetch_k: int = 20
    mmr_lambda: float = 0.3
    # Azure OpenAI
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    api_version_env: str = "AZURE_OPENAI_API_VERSION"
    endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    chat_deployment_env: str = "AZURE_OPENAI_CHAT_DEPLOYMENT"
    embed_deployment_env: str = "AZURE_OPENAI_EMBED_DEPLOYMENT"


SETTINGS = Settings()


# =========================
# Azure OpenAI components
# =========================

def get_azure_embeddings(settings: Settings) -> AzureOpenAIEmbeddings:
    """Create an Azure OpenAI Embeddings client from environment configuration.

    Parameters
    ----------
    settings : Settings
        Runtime configuration holding the environment variable names and defaults.

    Returns
    -------
    AzureOpenAIEmbeddings
        Configured embeddings client for computing vector representations.

    Raises
    ------
    RuntimeError
        If any of the required Azure OpenAI environment variables are missing.
    """
    endpoint = os.getenv(settings.endpoint_env)
    api_key = os.getenv(settings.api_key_env)
    api_version = os.getenv(settings.api_version_env)
    embed_deployment = os.getenv(settings.embed_deployment_env)

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
    """Instantiate an Azure Chat LLM client with deterministic generation settings.

    Parameters
    ----------
    settings : Settings
        Runtime configuration holding the environment variable names and defaults.

    Returns
    -------
    AzureChatOpenAI
        Configured chat LLM client used for answer generation and evaluation.

    Raises
    ------
    RuntimeError
        If any of the required Azure OpenAI environment variables are missing.
    """
    endpoint = os.getenv(settings.endpoint_env)
    api_key = os.getenv(settings.api_key_env)
    api_version = os.getenv(settings.api_version_env)
    chat_deployment = os.getenv(settings.chat_deployment_env)

    if not (endpoint and api_key and api_version and chat_deployment):
        raise RuntimeError(
            "Config Azure OpenAI mancante: assicurati di impostare ENDPOINT/API_KEY/API_VERSION e il deployment chat."
        )

    # temperature bassa per ridurre allucinazioni
    return AzureChatOpenAI(
        azure_deployment=chat_deployment,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
        temperature=0.0,
        max_tokens=800,
    )


# =========================
# Real documents loading
# =========================

def load_documents_from_dir(path: str | Path) -> List[Document]:
    """Load supported documents from a directory recursively.

    Supported extensions are ``.pdf``, ``.txt``, ``.md``, and optionally ``.docx``.

    Parameters
    ----------
    path : str or pathlib.Path
        Root directory to scan for documents.

    Returns
    -------
    list of langchain.schema.Document
        Loaded documents with basic metadata such as ``source``.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    RuntimeError
        If no supported documents are found in the directory.
    """
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

    # Aggiunge una sorgente leggibile per citazioni
    for i, d in enumerate(docs, start=1):
        d.metadata.setdefault("source", d.metadata.get("source", str(d.metadata.get("file_path", d.metadata.get("source", f"doc_{i}")))))
    return docs


def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """Split documents into overlapping chunks for retrieval.

    Parameters
    ----------
    docs : list of langchain.schema.Document
        Source documents to be chunked.
    settings : Settings
        Chunking configuration (size and overlap).

    Returns
    -------
    list of langchain.schema.Document
        Chunked documents preserving metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", "",
        ],
    )
    return splitter.split_documents(docs)


# =========================
# Vector store helpers (FAISS)
# =========================

def build_faiss_vectorstore(chunks: List[Document], embeddings: AzureOpenAIEmbeddings, persist_dir: str) -> FAISS:
    """Build and persist a FAISS vector store from document chunks.

    Parameters
    ----------
    chunks : list of langchain.schema.Document
        Pre-split document chunks to index.
    embeddings : AzureOpenAIEmbeddings
        Embeddings client used to embed the chunks.
    persist_dir : str
        Directory where the FAISS index and metadata will be saved.

    Returns
    -------
    FAISS
        The constructed FAISS vector store instance.
    """
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs


def load_or_build_vectorstore(settings: Settings, embeddings, docs: List[Document]) -> FAISS:
    """Load an existing FAISS vector store or build a new one.

    Parameters
    ----------
    settings : Settings
        Configuration including the persist directory path.
    embeddings : AzureOpenAIEmbeddings
        Embeddings client used to embed chunks.
    docs : list of langchain.schema.Document
        Raw documents to split and index if no persisted index exists.

    Returns
    -------
    FAISS
        Loaded or newly built vector store.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    if index_file.exists() and meta_file.exists():
        return FAISS.load_local(
            settings.persist_dir, embeddings, allow_dangerous_deserialization=True
        )
    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)


# def make_retriever(vector_store: FAISS, settings: Settings):
#     if settings.search_type == "mmr":
#         return vector_store.as_retriever(
#             search_type="mmr",
#             search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
#         )
#     return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": settings.k})


def make_retriever(vector_store: FAISS, settings: Settings):
    """Create a retriever with optional MMR and log retrieved chunks.

    Parameters
    ----------
    vector_store : FAISS
        The vector store to retrieve from.
    settings : Settings
        Retrieval configuration, such as ``search_type`` and ``k``.

    Returns
    -------
    object
        A retriever-like object exposing ``invoke`` and ``get_relevant_documents``
        methods that also prints a preview of retrieved chunks.
    """
    if settings.search_type == "mmr":
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.k,
                "fetch_k": settings.fetch_k,
                "lambda_mult": settings.mmr_lambda,
            },
        )
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )

    # Create a wrapper class that logs retrieved chunks
    class LoggingRetriever:
        def __init__(self, base_retriever):
            self.base_retriever = base_retriever
        
        def invoke(self, query: str):
            docs = self.base_retriever.invoke(query)
            print("\n=== Retrieved Chunks ===")
            for i, d in enumerate(docs, start=1):
                src = d.metadata.get("source", "N/A")
                print(f"\n--- Chunk {i} [source: {src}] ---\n{d.page_content[:500]}...\n")
            return docs
        
        def get_relevant_documents(self, query: str):
            # For backward compatibility
            return self.invoke(query)

    return LoggingRetriever(retriever)


        
def retrieve_chunks(question, retriever, max_chars: int = 500):
        """Retrieve top chunks for a query and return a compact, structured view.

        Parameters
        ----------
        question : str
            The user query used to retrieve relevant chunks.
        retriever : object
            Retriever exposing an ``invoke`` method returning a list of Documents.
        max_chars : int, optional
            Maximum number of characters to include from each chunk, by default 500.

        Returns
        -------
        list of dict
            A list of records with ``chunk_id``, ``source`` and truncated ``content``.
        """
        docs = retriever.invoke(question)

        results = []
        for i, d in enumerate(docs, start=1):
            results.append({
                "chunk_id": i,
                "source": d.metadata.get("source", "N/A"),
                "content": d.page_content[:max_chars] + ("..." if len(d.page_content) > max_chars else "")
            })
        return results



    #     def __init__(self, base_retriever):
    #         self.base_retriever = base_retriever
        
    #     def invoke(self, query: str):
    #         docs = self.base_retriever.invoke(query)
    #         print("\n=== Retrieved Chunks ===")
    #         for i, d in enumerate(docs, start=1):
    #             src = d.metadata.get("source", "N/A")
    #             print(f"\n--- Chunk {i} [source: {src}] ---\n{d.page_content[:500]}...\n")
    #         return docs
        
    #     def get_relevant_documents(self, query: str):
    #         # For backward compatibility
    #         return self.invoke(query)
        
    #     def __or__(self, other):
    #         # Support for LangChain pipeline operations
    #         from langchain_core.runnables import RunnableLambda
    #         return RunnableLambda(self.invoke) | other

    # return LoggingRetriever(retriever)


# =========================
# Prompting (qui: versione "gullible" richiesta)
# =========================

def format_docs_for_prompt(docs: List[Document]) -> str:
    """Format documents into a plain-text context block for prompting.

    Parameters
    ----------
    docs : list of langchain.schema.Document
        Documents to concatenate with ``[source:...]`` tags.

    Returns
    -------
    str
        A single string containing all documents with source annotations.
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)


def build_rag_chain(llm: AzureChatOpenAI, retriever) -> any:
    """Create a LangChain pipeline that retrieves context and generates answers.

    Parameters
    ----------
    llm : AzureChatOpenAI
        Chat LLM used for generation.
    retriever : object
        Retriever exposing an ``invoke`` method for fetching contextual documents.

    Returns
    -------
    Any
        A runnable chain that accepts a question string and returns a string answer.
    """
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

    from langchain_core.runnables import RunnableLambda
    
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


def rag_answer(question: str, chain) -> str:
    """Run the RAG chain on a question and return the answer text.

    Parameters
    ----------
    question : str
        The input question to answer.
    chain : Any
        Runnable chain created by :func:`build_rag_chain`.

    Returns
    -------
    str
        The generated answer.
    """
    return chain.invoke(question)


# =========================
# RAGAS helpers
# =========================

def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Return the text of top-k retrieved chunks for a question.

    Parameters
    ----------
    retriever : object
        Retriever exposing ``get_relevant_documents`` or ``invoke``.
    question : str
        The user question.
    k : int
        Number of top documents to include.

    Returns
    -------
    list of str
        The page contents of the retrieved documents.
    """
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
    """Run the RAG pipeline and build a dataset suitable for RAGAS evaluation.

    Parameters
    ----------
    questions : list of str
        Questions to evaluate.
    retriever : object
        Retriever used to fetch contexts.
    chain : Any
        Runnable chain producing answers.
    k : int
        Number of context chunks per question.
    ground_truth : dict, optional
        Optional mapping from question to reference answer.

    Returns
    -------
    list of dict
        Dataset entries with keys ``user_input``, ``retrieved_contexts``,
        ``response``, and optionally ``reference``.
    """
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


# =========================
# RAGAS evaluation utility
# =========================

def get_basketball_ground_truth() -> Dict[str, str]:
    """Return a small curated ground-truth QA set about basketball history.

    Returns
    -------
    dict
        Mapping from question to ground-truth answer.
    """
    return {
        "Who invented basketball?": "Dr. James Naismith.",
        "When was basketball invented?": "December 1891.",
        "Where was basketball invented?": "Springfield, Massachusetts, USA.",
        "Why did Naismith invent basketball?": "To keep students active indoors during winter.",
        "What organization was Naismith working for when he invented basketball?": "The YMCA Training School (now Springfield College).",
        "What objects were first used as baskets?": "Peach baskets.",
        "How many original rules did Naismith write?": "Thirteen.",
        "What was initially used as the ball in early basketball?": "A soccer ball.",
        "How did players originally score in early basketball?": "By throwing the ball into peach baskets nailed to a balcony.",
        "What problem did the original baskets have?": "They still had bottoms, so the ball had to be retrieved manually.",
    }


def evaluate_with_ragas(
    retriever,
    llm: AzureChatOpenAI,
    embeddings: AzureOpenAIEmbeddings,
    settings: Settings,
    questions: List[str],
    ground_truth: Optional[Dict[str, str]] = None,
):
    """Evaluate the pipeline with RAGAS and save a compact CSV of metrics.

    Parameters
    ----------
    retriever : object
        Retriever used for obtaining contexts.
    llm : AzureChatOpenAI
        LLM used by RAGAS where required.
    embeddings : AzureOpenAIEmbeddings
        Embeddings model used by RAGAS where required.
    settings : Settings
        Current configuration including retrieval parameters.
    questions : list of str
        Questions to evaluate.
    ground_truth : dict, optional
        Optional mapping from question to reference answers; enables
        the ``answer_correctness`` metric.

    Returns
    -------
    pandas.DataFrame
        A dataframe with normalized columns saved also to ``output/ragas_metrics.csv``.
    """
    chain = build_rag_chain(llm, retriever)

    dataset = build_ragas_dataset(
        questions=questions,
        retriever=retriever,
        chain=chain,
        k=settings.k,
        ground_truth=ground_truth,
    )

    evaluation_dataset = EvaluationDataset.from_list(dataset)
    ar=AnswerRelevancy(strictness=1)
    metrics = [context_precision, context_recall, faithfulness, ar]
    if ground_truth:
        metrics.append(answer_correctness)

    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    df = ragas_result.to_pandas()

    # Rimappa e normalizza colonne secondo richiesta utente
    rename_map = {
        "user_input": "question",
        "response": "answer",
        "reference": "ground_truth",
        # Nota: richiesta mantiene il refuso 'context_preicision'
        "context_precision": "context_preicision",
    }
    df_renamed = df.rename(columns=rename_map)

    requested_cols = [
        "question",
        "answer",
        "ground_truth",
        "context_preicision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    ]
    for col in requested_cols:
        if col not in df_renamed.columns:
            df_renamed[col] = pd.NA

    df_out = df_renamed[requested_cols]

    # Salva CSV con separatore ';' nella cartella output
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ragas_metrics.csv"
    df_out.to_csv(csv_path, sep=';', index=False)

    print("\n=== RAGAS metrics (k={}, search={}) ===".format(settings.k, settings.search_type))
    print(df_out.round(4).to_string(index=False))
    print(f"Saved RAGAS CSV to: {csv_path}")
    return df_out


# =========================
# main
# =========================

def rag_tool(question: str):
    """Retrieve top chunks and run a RAGAS evaluation over a fixed QA set.

    Parameters
    ----------
    question : str
        User question used to retrieve and preview relevant chunks.

    Returns
    -------
    list of dict
        Structured preview of retrieved chunks for the provided question.
    """
    settings = SETTINGS

    embeddings = get_azure_embeddings(settings)

    docs = load_documents_from_dir("./documenti")

    vector_store = load_or_build_vectorstore(settings, embeddings, docs)
    retriever = make_retriever(vector_store, settings)

    llm = get_azure_chat_llm(settings)

    # Retrieve chunks to return as tool output
    chunks = retrieve_chunks(question, retriever)

    # RAGAS evaluation on current question + curated GT set
    try:
        gt = get_basketball_ground_truth()
        # Evaluate only questions that have a ground-truth reference to avoid RAGAS schema errors
        eval_questions = [q for q in gt.keys()]
        evaluate_with_ragas(
            retriever=retriever,
            llm=llm,
            embeddings=embeddings,
            settings=settings,
            questions=eval_questions,
            ground_truth=gt,
        )
    except Exception as e:
        print(f"RAGAS evaluation skipped due to error: {e}")

    return chunks


if __name__ == "__main__":
    rag_tool("Example question for local run")



