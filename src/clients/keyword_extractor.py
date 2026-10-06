"""
Keyword Extractor.

Two-stage keyword extraction:
1. KeyBERT extracts initial keywords from text
2. LLM refines and picks best keywords for Semantic Scholar search
"""

import os
from typing import List, Tuple, Optional
from keybert import KeyBERT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.utils.config import settings
from src.utils.llm_factory import get_llm


class KeywordExtractor:
    """
    Extract and refine keywords for finding related papers.
    
    Two-stage process:
    1. KeyBERT: Fast embedding-based keyword extraction
    2. LLM: Refine keywords for better search results
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize extractor.
        
        Args:
            model_name: Sentence transformer model for KeyBERT
        """
        print(f"🔄 Loading KeyBERT model: {model_name}")
        self.keybert = KeyBERT(model_name)
        print("✅ KeyBERT model loaded")
        
        # Initialize LLM for refinement (uses config-based provider)
        self.llm = get_llm(temperature=0.2)
        
        self._setup_refiner()
    
    def _setup_refiner(self):
        """Set up LLM keyword refiner."""
        
        refine_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at identifying research topics for academic paper search.
Your task is to select and refine keywords that will find RELATED papers on Semantic Scholar."""),
            ("human", """Given this paper information and initial keywords, generate the BEST search keywords.

Paper Title: {title}
Paper Abstract: {abstract}

Initial Keywords (from KeyBERT):
{keywords}

Task:
1. Remove any URL fragments or meaningless terms (like "org corpus", "http", "github")
2. Keep only academically meaningful keywords
3. Add 2-3 broader research topic keywords that would find related work
4. Return exactly 8 keywords, comma-separated

Good keywords: "knowledge graph", "entity extraction", "citation network", "NLP"
Bad keywords: "org corpus", "github", "http", specific paper names

Return ONLY the 8 keywords, comma-separated, nothing else:""")
        ])
        
        self.refine_chain = refine_prompt | self.llm | StrOutputParser()
    
    def extract_raw_keywords(
        self, 
        text: str, 
        top_n: int = 15
    ) -> List[str]:
        """
        Extract keywords using KeyBERT only.
        
        Args:
            text: Text to extract keywords from
            top_n: Number of keywords
            
        Returns:
            List of keywords
        """
        keywords = self.keybert.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=top_n,
            use_mmr=True,  # Diverse keywords
            diversity=0.5
        )
        
        return [kw[0] for kw in keywords]
    
    def refine_keywords_with_llm(
        self,
        title: str,
        abstract: str,
        raw_keywords: List[str]
    ) -> List[str]:
        """
        Use LLM to refine and pick best keywords for search.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            raw_keywords: Keywords from KeyBERT
            
        Returns:
            Refined list of keywords
        """
        try:
            keywords_str = ", ".join(raw_keywords)
            
            result = self.refine_chain.invoke({
                "title": title,
                "abstract": abstract,
                "keywords": keywords_str
            })
            
            # Parse comma-separated keywords
            refined = [kw.strip() for kw in result.split(",")]
            refined = [kw for kw in refined if kw and len(kw) > 2]
            
            return refined[:8]  # Max 8 keywords
            
        except Exception as e:
            print(f"⚠️ LLM refinement failed: {e}")
            # Fallback: filter obvious bad keywords
            return self._basic_filter(raw_keywords)[:8]
    
    def _basic_filter(self, keywords: List[str]) -> List[str]:
        """Basic filter for obviously bad keywords."""
        bad_patterns = ['http', 'www', 'org', 'com', 'github', '.pdf', '/']
        filtered = []
        
        for kw in keywords:
            kw_lower = kw.lower()
            if not any(bad in kw_lower for bad in bad_patterns):
                filtered.append(kw)
        
        return filtered
    
    def extract_keywords(
        self,
        title: str,
        abstract: str,
        supplementary: Optional[str] = None,
        use_llm: bool = True
    ) -> List[str]:
        """
        Extract and refine keywords for search.
        
        This is the main method to use!
        
        Args:
            title: Paper title
            abstract: Paper abstract
            supplementary: Optional supplementary text to include
            use_llm: Whether to use LLM refinement
            
        Returns:
            List of refined keywords
        """
        # Combine title, abstract, and optional supplementary content
        parts = [title, abstract]
        if supplementary:
            parts.append(supplementary)
        text = ". ".join(parts)
        
        # Step 1: KeyBERT extraction
        raw_keywords = self.extract_raw_keywords(text, top_n=15)
        
        if not use_llm:
            return self._basic_filter(raw_keywords)[:8]
        
        # Step 2: LLM refinement
        refined_keywords = self.refine_keywords_with_llm(
            title=title,
            abstract=abstract,
            raw_keywords=raw_keywords
        )
        
        return refined_keywords


# === Singleton ===

_extractor = None

def get_keyword_extractor() -> KeywordExtractor:
    """Get singleton keyword extractor."""
    global _extractor
    if _extractor is None:
        _extractor = KeywordExtractor()
    return _extractor


# === Test ===

if __name__ == "__main__":
    print("="*60)
    print("KEYWORD EXTRACTOR TEST")
    print("="*60)
    
    # Test paper
    title = "Construction of the Literature Graph in Semantic Scholar"
    abstract = """We describe a deployed scalable system for organizing published 
    scientific literature into a heterogeneous graph to facilitate algorithmic 
    manipulation and discovery. The resulting literature graph consists of more 
    than 280M nodes, representing papers, authors, entities and various 
    interactions between them (e.g., authorships, citations, entity mentions). 
    We reduce literature graph construction into familiar NLP tasks (e.g., 
    entity extraction and linking), point out research challenges due to 
    differences from standard formulations of these tasks."""
    
    extractor = KeywordExtractor()
    
    # Step 1: Raw keywords
    print("\n🔑 RAW KEYWORDS (KeyBERT):")
    raw = extractor.extract_raw_keywords(f"{title}. {abstract}", top_n=10)
    for i, kw in enumerate(raw, 1):
        print(f"   [{i}] {kw}")
    
    # Step 2: Refined keywords
    print("\n✨ REFINED KEYWORDS (LLM):")
    refined = extractor.extract_keywords(title, abstract, use_llm=True)
    for i, kw in enumerate(refined, 1):
        print(f"   [{i}] {kw}")
    
    # Step 3: Search query
    print("\n🔍 SEARCH QUERY:")
    print(f"   '{' '.join(refined[:5])}'")
