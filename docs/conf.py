import os
import sys
from datetime import datetime

# Ensure the package under src/ is importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Project information
project = 'Agentic RAG'
author = 'Luca Battaglione'
copyright = f"{datetime.now().year}, {author}"
version = '0.1.0'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
]

# Extension configuration
autosummary_generate = True
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing
todo_include_todos = True

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css"
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# Output file base name for HTML help builder
htmlhelp_basename = 'AgenticRAGdoc'

# LaTeX output options
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files
latex_documents = [
    ('index', 'AgenticRAG.tex', 'Agentic RAG Documentation',
     'Luca Battaglione', 'manual'),
]

# Manual page output options
man_pages = [
    ('index', 'agenticrag', 'Agentic RAG Documentation',
     [author], 1)
]

# Texinfo output options
texinfo_documents = [
    ('index', 'AgenticRAG', 'Agentic RAG Documentation',
     author, 'AgenticRAG', 'Multi-agent RAG system with CrewAI and Qdrant.',
     'Miscellaneous'),
]

# Epub output options
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'langchain': ('https://api.python.langchain.com/en/latest/', None),
}

# ReadTheDocs-specific settings
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    # ReadTheDocs environment
    html_context = {
        "display_github": True,
        "github_user": "LucB21",
        "github_repo": "agentic_rag_qdrant",
        "github_version": "main",
        "conf_py_path": "/docs/",
    }
else:
    # Local build
    html_context = {}

# Source code links
html_show_sourcelink = True
html_copy_source = True
html_show_sphinx = True

# Search options
html_search_language = 'en'


