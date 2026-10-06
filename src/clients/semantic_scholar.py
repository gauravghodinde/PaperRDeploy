"""
Semantic Scholar API Client.

Fetches research papers from Semantic Scholar and ranks them
by citation count and relevance factors.
"""

import math
import time
import requests
from typing import List, Dict, Optional
from datetime import datetime

from src.models.paper import Paper, Author
from src.utils.config import settings


class SemanticScholarClient:
    """
    Client for Semantic Scholar API.
    
    Features:
    - Search papers by query
    - Get paper details by ID
    - Get paper references
    - Rank papers by quality (citations, recency, etc.)
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the client.
        
        Args:
            api_key: Optional API key for higher rate limits
        """
        self.api_key = api_key or settings.SEMANTIC_SCHOLAR_API_KEY
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "RelatedWorkGenerator/1.0"
        
        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> Dict:
        """
        Make API request with rate limiting.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            method: HTTP method
            
        Returns:
            JSON response
        """
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            if method == "GET":
                response = self.session.get(url, params=params)
            else:
                response = self.session.post(url, json=params)
            
            response.raise_for_status()
            
            # Rate limiting delay
            time.sleep(settings.API_DELAY_SECONDS)
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print(f"⚠️ Rate limited. Waiting 5 seconds...")
                time.sleep(5)
                return self._make_request(endpoint, params, method)
            raise e
    
    def search_papers(
        self,
        query: str,
        limit: int = 50,  # Fetch more, then filter to top 15
        year_from: Optional[int] = None,
        fields_of_study: Optional[List[str]] = None
    ) -> List[Paper]:
        """
        Search for papers by query.
        
        Args:
            query: Search query (keywords, title, etc.)
            limit: Maximum papers to fetch (before ranking)
            year_from: Only papers from this year onwards
            fields_of_study: Filter by research fields
            
        Returns:
            List of Paper objects
        """
        fields = [
            "paperId", "title", "abstract", "authors",
            "year", "venue", "citationCount", "referenceCount",
            "fieldsOfStudy", "openAccessPdf", "externalIds"
        ]
        
        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(fields)
        }
        
        if year_from:
            params["year"] = f"{year_from}-"
        
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        data = self._make_request("paper/search", params)
        papers = data.get("data", [])
        
        return [self._parse_paper(p) for p in papers if p.get("abstract")]
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """
        Get paper details by ID.
        
        Args:
            paper_id: Semantic Scholar ID, DOI, ArXiv ID, etc.
            
        Returns:
            Paper object or None
        """
        fields = [
            "paperId", "title", "abstract", "authors",
            "year", "venue", "citationCount", "referenceCount",
            "fieldsOfStudy", "openAccessPdf", "externalIds"
        ]
        
        params = {"fields": ",".join(fields)}
        
        try:
            data = self._make_request(f"paper/{paper_id}", params)
            return self._parse_paper(data)
        except Exception as e:
            print(f"⚠️ Could not fetch paper {paper_id}: {e}")
            return None
    
    def get_references(
        self, 
        paper_id: str, 
        limit: int = 50
    ) -> List[Paper]:
        """
        Get papers cited BY the given paper.
        
        Args:
            paper_id: Source paper ID
            limit: Maximum references to fetch
            
        Returns:
            List of referenced papers
        """
        fields = [
            "paperId", "title", "abstract", "authors",
            "year", "citationCount"
        ]
        
        params = {
            "limit": limit,
            "fields": ",".join(fields)
        }
        
        data = self._make_request(f"paper/{paper_id}/references", params)
        references = data.get("data", [])
        
        papers = []
        for ref in references:
            cited_paper = ref.get("citedPaper", {})
            if cited_paper and cited_paper.get("abstract"):
                papers.append(self._parse_paper(cited_paper))
        
        return papers
    
    def get_citations(
        self, 
        paper_id: str, 
        limit: int = 50
    ) -> List[Paper]:
        """
        Get papers that CITE the given paper.
        
        Args:
            paper_id: Target paper ID
            limit: Maximum citations to fetch
            
        Returns:
            List of citing papers
        """
        fields = [
            "paperId", "title", "abstract", "authors",
            "year", "citationCount"
        ]
        
        params = {
            "limit": limit,
            "fields": ",".join(fields)
        }
        
        data = self._make_request(f"paper/{paper_id}/citations", params)
        citations = data.get("data", [])
        
        papers = []
        for cit in citations:
            citing_paper = cit.get("citingPaper", {})
            if citing_paper and citing_paper.get("abstract"):
                papers.append(self._parse_paper(citing_paper))
        
        return papers
    
    def _parse_paper(self, data: Dict) -> Paper:
        """
        Parse API response into Paper model.
        
        Args:
            data: Raw API response
            
        Returns:
            Paper object
        """
        # Parse authors
        authors = []
        for author_data in data.get("authors", []):
            authors.append(Author(
                name=author_data.get("name", ""),
                author_id=author_data.get("authorId")
            ))
        
        return Paper(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract", "") or "",
            authors=authors,
            year=data.get("year"),
            venue=data.get("venue"),
            citation_count=data.get("citationCount", 0) or 0
        )
    
    def calculate_paper_score(self, paper: Paper) -> float:
        """
        Calculate quality score for ranking papers.
        
        Scoring factors:
        - Citation count (high weight, logarithmic)
        - Recency (papers from last 5 years get bonus)
        - Abstract quality (longer abstracts = more info)
        
        Args:
            paper: Paper to score
            
        Returns:
            Score (higher is better)
        """
        score = 0.0
        current_year = datetime.now().year
        
        # === Citation Score (max 50 points) ===
        # Logarithmic scale to prevent super-cited papers from dominating
        citations = paper.citation_count or 0
        if citations > 0:
            citation_score = min(50, math.log10(citations + 1) * 15)
            score += citation_score
        
        # === Recency Score (max 30 points) ===
        if paper.year:
            years_ago = current_year - paper.year
            if years_ago <= 2:
                score += 30  # Very recent
            elif years_ago <= 5:
                score += 25  # Recent
            elif years_ago <= 10:
                score += 15  # Relevant
            elif years_ago <= 15:
                score += 5   # Older
            # Older than 15 years: 0 recency points
        
        # === Abstract Quality (max 10 points) ===
        if paper.abstract:
            abstract_len = len(paper.abstract.split())
            if abstract_len >= 100:
                score += 10
            elif abstract_len >= 50:
                score += 5
        
        return score
    
    def get_top_papers(
        self,
        query: str,
        top_n: int = 15,
        fetch_limit: int = 50,
        year_from: Optional[int] = None,
        min_citations: int = 0
    ) -> List[Paper]:
        """
        Search and return TOP N papers ranked by quality.
        
        This is the main method to use!
        
        Args:
            query: Search query
            top_n: Number of top papers to return (default 15)
            fetch_limit: How many to fetch before ranking
            year_from: Only papers from this year onwards
            min_citations: Minimum citation count filter
            
        Returns:
            Top N papers sorted by quality score
        """
        print(f"🔍 Searching for: '{query}'")
        
        # Step 1: Fetch papers
        papers = self.search_papers(
            query=query,
            limit=fetch_limit,
            year_from=year_from
        )
        
        print(f"📄 Found {len(papers)} papers with abstracts")
        
        # Step 2: Filter by minimum citations
        if min_citations > 0:
            papers = [p for p in papers if p.citation_count >= min_citations]
            print(f"📊 {len(papers)} papers have >= {min_citations} citations")
        
        # Step 3: Calculate scores
        scored_papers = []
        for paper in papers:
            score = self.calculate_paper_score(paper)
            scored_papers.append((score, paper))
        
        # Step 4: Sort by score (descending)
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        
        # Step 5: Return top N
        top_papers = [paper for score, paper in scored_papers[:top_n]]
        
        print(f"⭐ Returning top {len(top_papers)} papers")
        
        # Print summary
        for i, paper in enumerate(top_papers, 1):
            score = scored_papers[i-1][0]
            print(f"  [{i}] {paper.title[:60]}...")
            print(f"      Year: {paper.year} | Citations: {paper.citation_count} | Score: {score:.1f}")
        
        return top_papers


# === Convenience Function ===

def search_related_papers(
    query: str,
    top_n: int = 15,
    year_from: Optional[int] = None
) -> List[Paper]:
    """
    Convenience function to search for related papers.
    
    Args:
        query: Search query
        top_n: Number of papers to return
        year_from: Year filter
        
    Returns:
        List of top papers
    """
    client = SemanticScholarClient()
    return client.get_top_papers(query, top_n=top_n, year_from=year_from)


# === Test Function ===

if __name__ == "__main__":
    # Quick test
    client = SemanticScholarClient()
    
    papers = client.get_top_papers(
        query="machine learning healthcare",
        top_n=15,
        year_from=2020
    )
    
    print("\n" + "="*60)
    print(f"TOP {len(papers)} PAPERS:")
    print("="*60)
    for i, paper in enumerate(papers, 1):
        print(f"\n[{i}] {paper.title}")
        print(f"    {paper.citation_key}")
        print(f"    Citations: {paper.citation_count}")
