from typing import List, Dict, Any, Optional, Tuple
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import quote_plus, urlencode
import time
from concurrent.futures import ThreadPoolExecutor
import logging
from fake_useragent import UserAgent
import re
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchResult:
    """Container for search results with validation."""
    def __init__(self, url: str, title: str, snippet: str, content: Optional[str] = None):
        self.url = url
        self.title = title
        self.snippet = snippet
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {
            'url': self.url,
            'title': self.title,
            'snippet': self.snippet,
            'content': self.content if self.content else self.snippet
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'SearchResult':
        return cls(
            url=data.get('url', ''),
            title=data.get('title', ''),
            snippet=data.get('snippet', ''),
            content=data.get('content')
        )

class DDGSearchAdapter:
    """Handles web search and content extraction using DuckDuckGo."""
    
    def __init__(self, max_results: int = 5, cache_size: int = 100):
        """
        Initialize DuckDuckGo search adapter.
        
        Args:
            max_results (int): Maximum number of search results to process
            cache_size (int): Size of the LRU cache for search results
        """
        self.max_results = max_results
        self.session = requests.Session()
        self.ua = UserAgent('Chrome')
        
        # Configure LRU cache for search results
        self.search_cache = lru_cache(maxsize=cache_size)(self._perform_search)
        
    def _extract_text(self, element: BeautifulSoup) -> str:
        """Safely extract text from a BeautifulSoup element."""
        return element.get_text(strip=True) if element else ''

    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid and safe to request."""
        if not url or not isinstance(url, str):
            return False
        
        # Check for unsafe schemes
        if re.match(r'^(javascript|data|file):', url, re.I):
            return False
            
        # Check for common web protocols
        if not re.match(r'^https?://', url, re.I):
            return False
            
        return True

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL by removing tracking parameters."""
        try:
            # Remove common tracking parameters
            pattern = r'utm_[a-zA-Z0-9]+=[^&]*&?'
            cleaned = re.sub(pattern, '', url)
            return cleaned.rstrip('?&')
        except Exception:
            return url

    def _perform_search(self, query: str) -> List[SearchResult]:
        """
        Perform actual DuckDuckGo search.
        
        Args:
            query (str): Search query
        
        Returns:
            List of SearchResult objects
        """
        url = "https://html.duckduckgo.com/html/"
        headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        try:
            response = self.session.post(
                url,
                headers=headers,
                data={'q': query, 'kl': 'us-en'},
                timeout=10
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for result in soup.select('.result'):
                try:
                    title_elem = result.select_one('.result__title a')
                    url = title_elem.get('href', '') if title_elem else ''
                    
                    if not self._is_valid_url(url):
                        continue
                        
                    url = self._normalize_url(url)
                    title = self._extract_text(title_elem)
                    snippet = self._extract_text(result.select_one('.result__snippet'))
                    
                    if url and title and snippet:
                        results.append(SearchResult(url, title, snippet))
                        
                        if len(results) >= self.max_results:
                            break
                            
                except Exception as e:
                    logger.debug(f"Error processing search result: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    def extract_content(self, search_result: SearchResult) -> Optional[str]:
        """
        Extract main content from a webpage.
        
        Args:
            search_result: SearchResult object containing URL to process
            
        Returns:
            Extracted content or None if extraction fails
        """
        if not self._is_valid_url(search_result.url):
            return None
            
        headers = {'User-Agent': self.ua.random}
        
        try:
            response = self.session.get(
                search_result.url,
                headers=headers,
                timeout=10,
                allow_redirects=True
            )
            response.raise_for_status()
            
            if 'text/html' not in response.headers.get('Content-Type', '').lower():
                return None
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for tag in ['script', 'style', 'nav', 'header', 'footer', 'iframe', 'aside']:
                for element in soup.find_all(tag):
                    element.decompose()
                    
            # Find main content
            main_content = None
            for selector in ['main', 'article', '[role="main"]', '#content', '.content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
                    
            if not main_content:
                main_content = soup
                
            # Extract meaningful paragraphs
            paragraphs = []
            for p in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = p.get_text().strip()
                if len(text) > 50 and not re.match(r'^[^a-zA-Z]*$', text):
                    paragraphs.append(text)
                    
            if not paragraphs:
                return None
                
            content = ' '.join(paragraphs)
            return content[:1000] if len(content) > 1000 else content
            
        except Exception as e:
            logger.error(f"Content extraction error for {search_result.url}: {e}")
            return None
            
    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Perform cached web search and process results.
        
        Args:
            query (str): Search query
        
        Returns:
            List of dictionaries containing processed search results
        """
        # Get cached search results
        search_results = self.search_cache(query)
        
        if not search_results:
            logger.warning(f"No search results found for query: {query}")
            return []
            
        processed_results = []
        
        # Process results in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_result = {
                executor.submit(self.extract_content, result): result
                for result in search_results
            }
            
            for future in future_to_result:
                result = future_to_result[future]
                try:
                    content = future.result()
                    if content:
                        result.content = content
                        processed_results.append(result.to_dict())
                    else:
                        # Use snippet as fallback if content extraction fails
                        result.content = result.snippet
                        processed_results.append(result.to_dict())
                except Exception as e:
                    logger.error(f"Failed to process {result.url}: {e}")
                    # Still include the result with just the snippet
                    result.content = result.snippet
                    processed_results.append(result.to_dict())
                    
        return processed_results
class ONIWithWebSearch(ONIWithRAG):
    def __init__(self, pdf_folder: str, max_retries: int = 3):
        super().__init__(pdf_folder)
        self.web_search = DDGSearchAdapter()
        self.max_retries = max_retries
        
    def _get_search_results(self, query: str) -> List[Dict[str, str]]:
        """Attempt to get search results with retries."""
        for attempt in range(self.max_retries):
            try:
                results = self.web_search.search(query)
                if results:
                    return results
                time.sleep(1 * (attempt + 1))
            except Exception as e:
                logger.error(f"Search attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1 * (attempt + 1))
        return []

    def run(self, text=None, image=None, show_thinking=False):
        if text is None:
            raise ValueError("`text` (user input) must be provided.")

        try:
            # Get relevant document chunks from RAG
            relevant_chunks = self.rag.query(text)

            # Perform web search with retries
            search_results = self._get_search_results(text)

            # Build enhanced context
            contexts = []
            
            if search_results:
                web_context = "\n".join([
                    f"From {result['url']}: {result['content']}" 
                    for result in search_results
                ])
                contexts.append(f"Web Search Results:\n{web_context}")
                print(f"Web Search Results:\n{web_context}")
            if relevant_chunks:
                local_context = "\n".join([
                    f"From {doc_id}: {chunk}" 
                    for doc_id, chunk, _ in relevant_chunks
                ])
                contexts.append(f"Local Context:\n{local_context}")

            # Combine original text with enhanced context
            enhanced_text = "\n\n".join(contexts + [f"Original Query: {text}"])

            # Call parent class's run method with enhanced text
            return super().run(enhanced_text, image, show_thinking)

        except Exception as e:
            logger.error(f"Error in ONIWithWebSearch.run: {e}")
            return "I encountered an error processing your request. Please try again."

pdf_folder = "PATH/ONI/knowledge_base/remembered_texts/test"
oni_with_search = ONIWithWebSearch(pdf_folder)
