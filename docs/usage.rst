Getting started
===============

Installation
------------

1. Ensure dependencies are installed (from the project root):

   - With uv/pip, or your environment manager of choice.

2. Set required Azure environment variables (example):

   - ``AZURE_OPENAI_API_KEY``
   - ``AZURE_OPENAI_API_VERSION``
   - ``AZURE_OPENAI_ENDPOINT``
   - ``AZURE_OPENAI_CHAT_DEPLOYMENT``
   - ``AZURE_OPENAI_EMBED_DEPLOYMENT``

Build the FAISS index and run the flow
-------------------------------------

Place your PDFs/TXT/MD/DOCX under ``documenti/``. The first run builds a
FAISS index under ``faiss_index_azure/``.

From the project root, run:

.. code-block:: bash

   python -m agentic_rag.main

You will be prompted for a question. The flow validates relevance and then
produces a RAG answer. Retrieved chunks and outputs are printed to the console.

Generate docs
-------------

From the project root:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in your browser.


