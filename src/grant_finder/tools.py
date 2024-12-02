# tools.py
from typing import Dict, List, Optional, Any, Type
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
from pydantic import Field, BaseModel

class CompanyDocumentTool(BaseTool):
    name: str = Field(default="company_document_tool", description="Tool for searching company documents")
    description: str = Field(default="Searches and analyzes company documents for relevant information")
    logger: Any = Field(default=None, description="Logger instance")
    doc_manager: Any = Field(default=None, description="Document manager instance")
    
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
        
class GrantCrawler(BaseTool):
    """Manages crawling of funding source websites"""
    name: str = Field(default="grant_crawler", description="Tool for crawling grant websites")
    description: str = Field(default="Crawls funding source websites to find grant opportunities")
    logger: Any = Field(default=None, description="Logger instance")
    visited_urls: set[str] = Field(default_factory=set, description="Set of URLs already visited")
    grant_pages: List[Dict] = Field(default_factory=list, description="List of found grant pages")
    
    def __init__(self, logger: logging.Logger):
        super().__init__()
        self.logger = logger
        self.visited_urls = set()
        self.grant_pages = []
    
    def _run(self, url: str) -> List[Dict]:
        """Required implementation of BaseTool._run"""
        return asyncio.run(self.crawl_site(url))
    
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
    name: str = Field(default="enhanced_grant_search", description="Tool for comprehensive grant searching")
    description: str = Field(default="Performs comprehensive search across multiple grant sources and databases")
    logger: Any = Field(default=None, description="Logger instance")
    serp_api_key: str = Field(default=None, description="SerpAPI key")
    funding_sources_path: Path = Field(default=None, description="Path to funding sources file")
    cache_dir: Path = Field(default=None, description="Cache directory path")
    crawler: Any = Field(default=None, description="Grant crawler instance")
    search_wrapper: Any = Field(default=None, description="Search wrapper instance")
    funding_sources: Dict = Field(default_factory=dict, description="Funding sources data")
    
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
        """Required implementation of BaseTool._run"""
        try:
            results = []
            
            # 1. Crawl funding source websites
            sources_to_crawl = [
                source for source in self.funding_sources.values()
                if (not source['last_crawled'] or 
                    datetime.now() - source['last_crawled'] > source['crawl_frequency'])
            ]
            
            if sources_to_crawl:
                for source in sources_to_crawl:
                    crawl_results = self.crawler._run(source['url'])
                    results.extend(crawl_results)
            
            # 2. Search cached crawl results
            cached_results = self._search_cached_results(query)
            results.extend(cached_results)
            
            # 3. Perform web search
            web_results = self._web_search(query)
            results.extend(web_results)
            
            # 4. Deduplicate and rank results
            final_results = self._process_results(results, query)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Grant search failed: {str(e)}")
            raise GrantSearchError(f"Grant search failed: {str(e)}")
    
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
    name: str = Field(default="web_scrape", description="Tool for web scraping")
    description: str = Field(default="Scrapes web pages for grant information")
    logger: Any = Field(default=None, description="Logger instance")
    
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
    name: str = Field(default="funding_source_manager", description="Manages and tracks funding sources")
    description: str = Field(default="Manages and tracks funding sources")
    logger: Any = Field(default=None, description="Logger instance")
    file_path: Path = Field(default=None, description="Path to funding sources file")
    
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
    

class StrategyRequirementsTool(BaseTool):
    name: str = Field(default="strategy_requirements_tool", description="Tool for developing strategic search requirements")
    description: str = Field(
        default="""Tool for developing strategic search requirements that align company capabilities with grant opportunities. 
        Input should be a JSON string containing company profile information.
        Output will be detailed strategic requirements for grant searching."""
    )
    
    def _run(self, query: str) -> str:
        """
        Process company information to develop strategic search requirements.
        
        Args:
            query (str): JSON string containing company profile data and focus areas
            
        Returns:
            str: JSON string containing structured strategic requirements
        """
        try:
            # Parse input data
            company_data = json.loads(query)
            
            # Extract key capabilities from technical focus and experience
            capabilities = self._extract_capabilities(
                company_data.get("technical_focus", ""),
                company_data.get("technical_experience", "")
            )
            
            # Identify innovation areas from capabilities and company vision
            innovation_areas = self._identify_innovation_areas(
                capabilities,
                company_data.get("vision", ""),
                company_data.get("innovations", "")
            )
            
            # Determine competitive advantages
            competitive_advantages = self._analyze_competitive_advantages(
                company_data.get("sbir_experience", ""),
                capabilities,
                innovation_areas
            )
            
            # Determine appropriate target phases
            target_phases = self._determine_target_phases(
                company_data.get("sbir_experience", "")
            )
            
            # Build structured requirements
            requirements = {
                "technical_requirements": [
                    {
                        "requirement": cap,
                        "justification": "Derived from company's demonstrated technical capabilities"
                    } for cap in capabilities
                ],
                "innovation_areas": [
                    {
                        "area": area,
                        "alignment": "Strong alignment with company's strategic focus"
                    } for area in innovation_areas
                ],
                "competitive_advantages": [
                    {
                        "advantage": adv,
                        "evidence": "Based on documented company experience"
                    } for adv in competitive_advantages
                ],
                "target_phases": target_phases
            }
            
            return json.dumps(requirements, indent=2)
            
        except json.JSONDecodeError:
            raise ValueError("Input must be a valid JSON string containing company profile data")
        except Exception as e:
            raise RuntimeError(f"Error processing strategic requirements: {str(e)}")
    
    def _extract_capabilities(self, technical_focus: str, technical_experience: str) -> List[str]:
        """Extract specific technical capabilities"""
        capabilities = set()
        
        # Break down technical focus into specific capabilities
        if technical_focus:
            focus_areas = [focus.strip() for focus in technical_focus.split(',')]
            for area in focus_areas:
                if "AI" in area or "ML" in area:
                    capabilities.add("Deep learning model development for specific applications")
                    capabilities.add("ML algorithm optimization for resource-constrained environments")
                if "data" in area.lower():
                    capabilities.add("Real-time data processing and analytics")
                    capabilities.add("Large-scale data integration and management")
        
        # Extract capabilities from experience
        if technical_experience:
            if "deployment" in technical_experience.lower():
                capabilities.add("Production system deployment and scaling")
            if "research" in technical_experience.lower():
                capabilities.add("Novel algorithm development and implementation")
        
        return list(capabilities)
    
    def _identify_innovation_areas(self, capabilities: List[str], vision: str, innovations: str) -> List[str]:
        """Identify specific innovation areas"""
        innovation_areas = set()
        
        # Map capabilities to innovation areas
        capability_to_innovation = {
            "Deep learning": "Applied AI for mission-critical systems",
            "data processing": "Real-time analytics for operational systems",
            "algorithm": "Novel computational approaches",
        }
        
        # Add relevant innovation areas based on capabilities
        for cap in capabilities:
            for key, innovation in capability_to_innovation.items():
                if key.lower() in cap.lower():
                    innovation_areas.add(innovation)
        
        # Add areas from company innovations
        if innovations:
            innovation_lines = innovations.split('\n')
            for line in innovation_lines:
                if line.strip():
                    innovation_areas.add(line.strip())
        
        return list(innovation_areas)
    
    def _analyze_competitive_advantages(
        self, 
        sbir_experience: str, 
        capabilities: List[str],
        innovation_areas: List[str]
    ) -> List[str]:
        """Determine competitive advantages"""
        advantages = set()
        
        # Add advantages based on SBIR experience
        if sbir_experience:
            if "Phase II" in sbir_experience:
                advantages.add("Demonstrated success in Phase II SBIR/STTR programs")
            if "Phase I" in sbir_experience:
                advantages.add("Track record of Phase I SBIR/STTR awards")
        
        # Add technical advantages
        for cap in capabilities:
            advantages.add(f"Proven expertise in {cap}")
        
        # Add innovation-based advantages
        for area in innovation_areas:
            advantages.add(f"Leading innovation in {area}")
        
        return list(advantages)
    
    def _determine_target_phases(self, sbir_experience: str) -> List[Dict[str, str]]:
        """Determine appropriate SBIR/STTR phases to target"""
        phases = []
        
        # Always include Phase I
        phases.append({
            "phase": "Phase I",
            "rationale": "Initial R&D feasibility studies"
        })
        
        # Add Phase II if experienced
        if "Phase I" in sbir_experience:
            phases.append({
                "phase": "Phase II",
                "rationale": "Full R&D effort building on Phase I success"
            })
        
        # Add Direct to Phase II if highly experienced
        if "Phase II" in sbir_experience:
            phases.append({
                "phase": "Direct to Phase II",
                "rationale": "Leveraging prior work demonstrating Phase I feasibility"
            })
        
        return phases
    
    async def _arun(self, query: str) -> str:
        """Async implementation"""
        return self._run(query)
    

class OpportunityEvent(BaseModel):
    """Model for events related to funding opportunities"""
    title: str
    type: str  # "ProposerDay", "AMA", "RFI", etc.  
    date: str
    location: str
    description: str
    registration_deadline: Optional[str]
    registration_link: Optional[str]
    related_topics: List[str] = Field(default_factory=list)
    
class FundingTopic(BaseModel):
    """Model for specific funding topics"""
    topic_id: str
    title: str 
    description: str
    phase_1_objectives: Optional[List[str]]
    phase_2_objectives: Optional[List[str]]
    award_amount: str
    submission_deadline: str
    agency: str
    program: str
    baa_number: Optional[str]
    related_events: List[str] = Field(default_factory=list)
    dod_challenges: List[str] = Field(default_factory=list)

class StrategicPlan(BaseModel):
    """Model for the strategic pursuit plan"""
    immediate_actions: List[Dict[str, str]]
    thirty_day_actions: List[Dict[str, str]]
    sixty_day_actions: List[Dict[str, str]]
    ninety_day_actions: List[Dict[str, str]]
    proposal_schedule: List[Dict[str, str]]
    event_schedule: List[Dict[str, str]]
    information_gaps: List[str]
    recommended_partnerships: List[Dict[str, str]]

class StrategicPlannerTool(BaseTool):
    name = "strategic_planner_tool"
    description = """Tool for developing comprehensive strategic plans for pursuing funding opportunities.
    Analyzes BAAs, specific topics, events, and deadlines to create actionable pursuit strategies."""
    
    def __init__(self):
        super().__init__()
        self.current_date = datetime.now()
    
    def _is_date_valid(self, date_str: str) -> bool:
        """Check if a date is in the future"""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            return date > self.current_date
        except:
            return False
            
    def _organize_opportunities(self, data: dict) -> Dict[str, List]:
        """Organize opportunities by type and deadline"""
        organized = {
            "active_baas": [],
            "specific_topics": [],
            "upcoming_events": [],
            "dod_challenges": [],
            "information_requests": []
        }
        
        # Process each opportunity
        for opp in data.get("opportunities", []):
            # Skip if deadline has passed
            if not self._is_date_valid(opp.get("submission_deadline", "1900-01-01")):
                continue
                
            if "BAA" in opp.get("topic_id", ""):
                organized["active_baas"].append(opp)
            elif opp.get("topic_id"):
                organized["specific_topics"].append(opp)
                
        # Process events
        for event in data.get("events", []):
            if not self._is_date_valid(event.get("date", "1900-01-01")):
                continue
                
            event_type = event.get("type", "").lower()
            if "rfi" in event_type:
                organized["information_requests"].append(event)
            else:
                organized["upcoming_events"].append(event)
                
        # Process DoD challenges
        for challenge in data.get("dod_challenges", []):
            if self._is_date_valid(challenge.get("submission_deadline", "1900-01-01")):
                organized["dod_challenges"].append(challenge)
                
        return organized

    def _develop_strategic_plan(self, 
        organized_opps: Dict[str, List],
        company_profile: dict,
        search_requirements: dict
    ) -> StrategicPlan:
        """Develop strategic plan based on opportunities and company profile"""
        
        # Initialize plan components
        plan = StrategicPlan(
            immediate_actions=[],
            thirty_day_actions=[],
            sixty_day_actions=[],
            ninety_day_actions=[],
            proposal_schedule=[],
            event_schedule=[],
            information_gaps=[],
            recommended_partnerships=[]
        )
        
        # Process specific topics for proposal schedule
        for topic in organized_opps["specific_topics"]:
            deadline = datetime.strptime(topic["submission_deadline"], '%Y-%m-%d')
            days_until = (deadline - self.current_date).days
            
            schedule_item = {
                "topic_id": topic["topic_id"],
                "title": topic["title"],
                "deadline": topic["submission_deadline"],
                "agency": topic["agency"],
                "priority": "High" if days_until < 45 else "Medium"
            }
            plan.proposal_schedule.append(schedule_item)
            
            # Add immediate actions for close deadlines
            if days_until < 30:
                plan.immediate_actions.append({
                    "action": f"Begin proposal preparation for {topic['topic_id']}",
                    "deadline": topic["submission_deadline"],
                    "type": "Proposal"
                })
            
        # Schedule event participation
        for event in organized_opps["upcoming_events"]:
            event_date = datetime.strptime(event["date"], '%Y-%m-%d')
            days_until = (event_date - self.current_date).days
            
            schedule_item = {
                "event": event["title"],
                "date": event["date"],
                "type": event["type"],
                "priority": "High" if event["related_topics"] else "Medium"
            }
            plan.event_schedule.append(schedule_item)
            
            # Add registration to appropriate action timeline
            if days_until < 30:
                plan.immediate_actions.append({
                    "action": f"Register for {event['title']}",
                    "deadline": event["registration_deadline"],
                    "type": "Event"
                })
            elif days_until < 60:
                plan.thirty_day_actions.append({
                    "action": f"Register for {event['title']}",
                    "deadline": event["registration_deadline"],
                    "type": "Event"
                })
                
        # Check for information gaps
        required_capabilities = set()
        for topic in organized_opps["specific_topics"]:
            if "phase_2_objectives" in topic:
                required_capabilities.update(topic["phase_2_objectives"])
                
        company_capabilities = set(company_profile.get("technical_focus", "").split(","))
        capability_gaps = required_capabilities - company_capabilities
        
        if capability_gaps:
            plan.information_gaps.extend([
                f"Missing capability: {cap}" for cap in capability_gaps
            ])
            
        # Recommend partnerships for capability gaps
        for gap in capability_gaps:
            plan.recommended_partnerships.append({
                "capability_needed": gap,
                "partnership_type": "Technical",
                "priority": "High"
            })
            
        return plan

    def _run(self, query: str) -> str:
        """
        Process opportunities and develop strategic plan
        
        Args:
            query (str): JSON string containing opportunities, events, company profile, and search requirements
            
        Returns:
            str: JSON string containing strategic plan
        """
        try:
            # Parse input data
            data = json.loads(query)
            
            # Organize opportunities
            organized_opps = self._organize_opportunities(data)
            
            # Develop strategic plan
            strategic_plan = self._develop_strategic_plan(
                organized_opps,
                data.get("company_profile", {}),
                data.get("search_requirements", {})
            )
            
            return json.dumps(strategic_plan.dict(), indent=2)
            
        except json.JSONDecodeError:
            raise ValueError("Input must be a valid JSON string")
        except Exception as e:
            raise RuntimeError(f"Error developing strategic plan: {str(e)}")
    
    async def _arun(self, query: str) -> str:
        """Async implementation"""
        return self._run(query)

class FinalReportTool(BaseTool):
    name = "final_report_tool"
    description = """Tool for generating comprehensive grant analysis reports.
    Use this tool to analyze opportunities and create strategic recommendations."""
    
    def _run(self, query: str) -> str:
        """Generate report analysis"""
        # Implement actual report generation logic here
        return f"Generated analysis report for: {query}"
        
    async def _arun(self, query: str) -> str:
        """Async implementation"""
        return self._run(query)