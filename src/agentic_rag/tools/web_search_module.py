"""
Web Search Tool for RAG Enhancement

This tool provides web search capabilities to complement the RAG system
with up-to-date information from the internet. It respects the trust system
by checking web domains against the trusted domains whitelist.
"""

from typing import Dict, List, Any
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse
import time
from crewai_tools import BaseTool
from pydantic import BaseModel, Field


class WebSearchTool(BaseTool):
    """Tool for searching the web and extracting relevant information."""
    
    name: str = "web_search"
    description: str = (
        "Search the web for information related to a given query. "
        "Returns a list of relevant web results with titles, URLs, and content snippets. "
        "Applies trust classification based on domain whitelist."
    )
    
    # Default trusted domains (can be overridden)
    trusted_domains: List[str] = [
        "wikipedia.org",
        "gov",
        "edu",
        "britannica.com",
        "nature.com",
        "sciencedirect.com",
        "consob.it",
        "ey.net"
    ]
    
    def _classify_domain_trust(self, url: str) -> str:
        """
        Classify domain trust level based on whitelist.
        
        Args:
            url: The URL to classify
            
        Returns:
            "trusted" if domain is in whitelist, "untrusted" otherwise
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Check if domain is in trusted list
            is_trusted = any(trusted_domain.lower() in domain for trusted_domain in self.trusted_domains)
            
            return "trusted" if is_trusted else "untrusted"
            
        except Exception:
            return "untrusted"  # Default to untrusted if parsing fails
    
    def _run(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform a web search using DuckDuckGo.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing search results
        """
        try:
            results = self._search_duckduckgo(query, max_results)
            return {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "results": [],
                "count": 0
            }
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """
        Search DuckDuckGo for the given query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # DuckDuckGo search URL
        search_url = f"https://duckduckgo.com/html/?q={quote(query)}"
        
        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Find search result elements
            result_elements = soup.find_all('div', class_='result')
            
            for element in result_elements[:max_results]:
                try:
                    # Extract title
                    title_elem = element.find('a', class_='result__a')
                    title = title_elem.text.strip() if title_elem else "No title"
                    
                    # Extract URL
                    url = title_elem.get('href', '') if title_elem else ''
                    
                    # Extract snippet
                    snippet_elem = element.find('div', class_='result__snippet')
                    snippet = snippet_elem.text.strip() if snippet_elem else "No snippet available"
                    
                    if title and url:
                        # Classify trust level based on domain whitelist
                        trust_level = self._classify_domain_trust(url)
                        
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "source_type": "web",
                            "trust_level": trust_level  # Based on domain whitelist
                        })
                        
                except Exception as e:
                    continue
                    
            return results
            
        except Exception as e:
            print(f"Error searching DuckDuckGo: {e}")
            return []
    
    def _extract_content(self, url: str, max_chars: int = 2000) -> str:
        """
        Extract content from a web page.
        
        Args:
            url: URL to extract content from
            max_chars: Maximum characters to extract
            
        Returns:
            Extracted text content
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:max_chars] if len(text) > max_chars else text
            
        except Exception as e:
            return f"Error extracting content from {url}: {str(e)}"


# Create an instance of the web search tool
web_search_tool = WebSearchTool()