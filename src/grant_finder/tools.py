# tools.py
from typing import Dict, List, Optional, Any
from langchain.tools import BaseTool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from bs4 import BeautifulSoup
import requests
import csv
import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
from pathlib import Path
from .document_store import DocumentStoreManager
from .types import DocumentSearchResult, WebSearchResult, GrantSearchError

class CompanyDocumentTool(BaseTool):
    def __init__(self, directory_path: str, logger: logging.Logger):
        super().__init__()
        self.logger = logger
        docs_dir = Path(directory_path)
        storage_dir = docs_dir / ".document_store"
        
        self.doc_manager = DocumentStoreManager(
            docs_dir=docs_dir,
            storage_dir=storage_dir
        )
        
        # Update indices on initialization
        new_docs = self.doc_manager.update_document_index()
        if new_docs:
            self.logger.info(f"Indexed {len(new_docs)} new or modified documents")
    
    def _run(self, query: str) -> List[DocumentSearchResult]:
        try:
            documents = self.doc_manager.get_relevant_documents(query)
            return [
                DocumentSearchResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    relevance_score=1.0  # Score from vector similarity
                )
                for doc in documents
            ]
        except Exception as e:
            self.logger.error(f"Document search failed: {str(e)}")
            raise GrantSearchError(f"Document search failed: {str(e)}")
        
class GrantCrawler:
    """Manages crawling of funding source websites"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.visited_urls = set()
        self.grant_pages = []
        
    async def crawl_site(self, base_url: str, depth: int = 2) -> List[Dict]:
        """Crawl a funding source website to specified depth"""
        self.visited_urls.clear()
        self.grant_pages.clear()
        
        async with aiohttp.ClientSession() as session:
            await self._crawl_page(session, base_url, depth)
            
        return self.grant_pages
    
    async def _crawl_page(self, session: aiohttp.ClientSession, url: str, depth: int):
        """Recursively crawl pages looking for grant content"""
        if depth <= 0 or url in self.visited_urls:
            return
            
        self.visited_urls.add(url)
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return
                    
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Check if this is a grant page
                if self._is_grant_page(soup):
                    self.grant_pages.append({
                        'url': url,
                        'title': self._extract_title(soup),
                        'content': self._extract_content(soup),
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Find links to follow
                if depth > 0:
                    links = self._extract_relevant_links(soup, url)
                    for link in links:
                        await self._crawl_page(session, link, depth - 1)
                        
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {str(e)}")
    
    def _is_grant_page(self, soup: BeautifulSoup) -> bool:
        """Determine if page contains grant information"""
        text = soup.get_text().lower()
        keywords = ['grant', 'funding', 'solicitation', 'proposal', 'sbir', 'sttr']
        return any(keyword in text for keyword in keywords)
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract grant title from page"""
        title_tag = soup.find('h1') or soup.find('title')
        return title_tag.get_text().strip() if title_tag else ""
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from page"""
        # Remove navigation, headers, footers
        for tag in soup.find_all(['nav', 'header', 'footer']):
            tag.decompose()
        
        return soup.get_text(separator=' ', strip=True)
    
    def _extract_relevant_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract relevant links to follow"""
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(base_url, href)
            
            # Only follow links from same domain
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                links.append(full_url)
                
        return links

class EnhancedGrantSearchTool(BaseTool):
    """Enhanced tool for comprehensive grant searching"""
    
    def __init__(
        self, 
        serp_api_key: str,
        funding_sources_path: str,
        logger: logging.Logger,
        cache_dir: Optional[Path] = None
    ):
        super().__init__()
        self.serp_api_key = serp_api_key
        self.funding_sources_path = Path(funding_sources_path)
        self.logger = logger
        self.cache_dir = cache_dir or Path("cache/grant_search")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.crawler = GrantCrawler(logger)
        self.search_wrapper = SerpAPIWrapper(serpapi_api_key=serp_api_key)
        
        # Load funding sources
        self.funding_sources = self._load_funding_sources()
    
    def _load_funding_sources(self) -> Dict:
        """Load and validate funding sources"""
        sources = {}
        with open(self.funding_sources_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Title'] and row['Source']:
                    sources[row['Title']] = {
                        'url': row['Source'],
                        'last_crawled': None,
                        'crawl_frequency': timedelta(days=1)  # Configurable
                    }
        return sources
    
    def _run(self, query: str) -> List[Dict]:
        """Run comprehensive grant search"""
        results = []
        
        # 1. Crawl funding source websites
        sources_to_crawl = [
            source for source in self.funding_sources.values()
            if (not source['last_crawled'] or 
                datetime.now() - source['last_crawled'] > source['crawl_frequency'])
        ]
        
        if sources_to_crawl:
            asyncio.run(self._crawl_sources(sources_to_crawl))
        
        # 2. Search cached crawl results
        cached_results = self._search_cached_results(query)
        results.extend(cached_results)
        
        # 3. Perform web search
        web_results = self._web_search(query)
        results.extend(web_results)
        
        # 4. Deduplicate and rank results
        final_results = self._process_results(results, query)
        
        return final_results
    
    async def _crawl_sources(self, sources: List[Dict]):
        """Crawl multiple funding sources concurrently"""
        tasks = []
        for source in sources:
            task = asyncio.create_task(
                self.crawler.crawl_site(source['url'])
            )
            tasks.append(task)
            
        crawl_results = await asyncio.gather(*tasks)
        
        # Cache results
        timestamp = datetime.now().isoformat()
        cache_file = self.cache_dir / f"crawl_{timestamp}.json"
        with open(cache_file, 'w') as f:
            json.dump(crawl_results, f)
    
    def _search_cached_results(self, query: str) -> List[Dict]:
        """Search through cached crawl results"""
        results = []
        for cache_file in self.cache_dir.glob("crawl_*.json"):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                for item in cached_data:
                    if self._matches_query(item, query):
                        results.append(item)
        return results
    
    def _web_search(self, query: str) -> List[Dict]:
        """Perform general web search"""
        enhanced_query = f"SBIR STTR grant funding {query}"
        raw_results = self.search_wrapper.run(enhanced_query)
        
        return [{
            'url': result['link'],
            'title': result['title'],
            'content': result['snippet'],
            'source': 'web_search',
            'timestamp': datetime.now().isoformat()
        } for result in raw_results.get('organic', [])]
    
    def _matches_query(self, item: Dict, query: str) -> bool:
        """Check if cached item matches search query"""
        query_terms = query.lower().split()
        text = f"{item['title']} {item['content']}".lower()
        return all(term in text for term in query_terms)
    
    def _process_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Deduplicate and rank results"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                result['relevance_score'] = self._calculate_relevance(result, query)
                unique_results.append(result)
        
        # Sort by relevance
        unique_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return unique_results[:10]  # Return top 10 results
    
    def _calculate_relevance(self, result: Dict, query: str) -> float:
        """Calculate result relevance score"""
        query_terms = query.lower().split()
        text = f"{result['title']} {result['content']}".lower()
        
        # Simple scoring based on term presence and position
        score = 0
        for term in query_terms:
            if term in text:
                score += 1
                if term in result['title'].lower():
                    score += 0.5
        
        return score / len(query_terms)

class WebScrapeTool(BaseTool):
    name = "web_content_scraper"
    description = "Scrapes and extracts content from grant websites"
    
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.logger = logger
    
    def _run(self, url: str) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style']):
                element.decompose()
            
            # Extract main content
            content = " ".join(soup.stripped_strings)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Web scraping failed for {url}: {str(e)}")
            raise GrantSearchError(f"Web scraping failed: {str(e)}")

class FundingSourceTool(BaseTool):
    name = "funding_source_manager"
    description = "Manages and tracks funding sources"
    
    def __init__(self, file_path: str, logger: logging.Logger):
        super().__init__()
        self.file_path = Path(file_path)
        self.logger = logger
    
    def _run(self, action: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            if action == "load":
                return self._load_sources()
            elif action == "update":
                return self._update_source(data)
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Funding source operation failed: {str(e)}")
            raise GrantSearchError(f"Funding source operation failed: {str(e)}")
    
    def _load_sources(self) -> Dict[str, Any]:
        import csv
        sources = {}
        
        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Title'] and row['Source']:
                    sources[row['Title']] = {
                        "url": row['Source'],
                        "was_searched": False,
                        "search_successful": False,
                        "grants_found": []
                    }
        
        return sources
    
    def _update_source(self, data: Dict) -> Dict[str, Any]:
        if not data or 'name' not in data:
            raise ValueError("Invalid update data")
            
        sources = self._load_sources()
        if data['name'] in sources:
            sources[data['name']].update(data)
        
        return sources