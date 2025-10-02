# Agentic RAG with Qdrant

An advanced multi-agent Retrieval-Augmented Generation (RAG) system powered by [crewAI](https://crewai.com) and [Qdrant](https://qdrant.tech/). This project implements an intelligent document retrieval system with web search capabilities, leveraging multiple AI agents to provide comprehensive and accurate responses to user queries.

## Features

- **Multi-Agent Architecture**: Intelligent check crew, RAG crew, and synthesis crew working together
- **Vector Database**: Qdrant for efficient document storage and retrieval
- **Web Search Integration**: Automatic web search when local documents are insufficient
- **RAGAS Evaluation**: Comprehensive evaluation metrics for RAG performance
- **Streamlit Interface**: User-friendly web interface
- **Docker Support**: Easy deployment with Docker Compose

## Quick Start

### Prerequisites

- Python >=3.10 <3.14
- [UV](https://docs.astral.sh/uv/) for dependency management
- Azure OpenAI API key
- Qdrant instance (local or cloud)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LucB21/agentic_rag_qdrant.git
cd agentic_rag_qdrant
```

2. Install UV (if not already installed):
```bash
pip install uv
```

3. Install dependencies:
```bash
uv sync
```

4. Set up environment variables in `.env`:
```bash
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_VERSION=2024-02-01
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key  # if using cloud
```

### Running with Docker

1. Start the services:
```bash
docker-compose up -d
```

2. Access the Streamlit app at `http://localhost:8501`

### Running Locally

1. Start Qdrant (if using local instance):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Installation Guide](docs/installation.rst)**: Detailed setup instructions
- **[Quick Start](docs/quickstart.rst)**: Get up and running quickly
- **[Configuration](docs/configuration.rst)**: Configuration options
- **[Architecture](docs/architecture/)**: System architecture and design
- **[Evaluation](docs/evaluation/)**: RAGAS evaluation setup and metrics
- **[Examples](docs/examples/)**: Usage examples and tutorials
- **[Deployment](docs/deployment/)**: Production deployment guides

### Building Documentation

To build the documentation locally:

```bash
cd docs
sphinx-build -b html . _build/html
```

The documentation will be available at `docs/_build/html/index.html`.

## System Architecture

The system consists of three main crews:

1. **Check Crew**: Analyzes user queries to determine relevance to the document corpus
2. **RAG Crew**: Performs vector search against Qdrant database when documents are relevant
3. **Web Search Crew**: Conducts web searches when local documents are insufficient
4. **Synthesis Crew**: Combines and synthesizes information from multiple sources

## Evaluation

The system includes comprehensive evaluation using RAGAS metrics:

- **Context Precision**: Measures relevance of retrieved contexts
- **Context Recall**: Measures completeness of retrieved contexts
- **Faithfulness**: Measures factual accuracy of generated responses
- **Answer Relevancy**: Measures relevance of answers to questions

Run evaluation:
```bash
cd test_ragas
python rag_ragas_qdrant.py
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, questions, or feedback:
- Create an issue on GitHub
- Check the [documentation](docs/)
- Review the examples in `docs/examples/`

- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
