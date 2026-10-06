"""
Summary Chain.

Summarizes research papers using LangChain and Groq (FREE tier).
Generates concise 40-50 word summaries for each paper.
"""

import os
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.models.paper import Paper, Summary
from src.utils.config import settings
from src.utils.llm_factory import get_llm


class SummaryChain:
    """
    Generate summaries for research papers using LLM.
    
    Takes paper abstract and generates a concise summary.
    """
    
    def __init__(self):
        """
        Initialize summary chain.
        """
        # Use config-based LLM provider
        self.llm = get_llm(temperature=0.3)
        
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the summarization chain."""
        
        # Summary prompt - generates concise 40-50 word summary
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at summarizing research papers.
Your task is to create concise, informative summaries that capture the key contribution of each paper."""),
            ("human", """Summarize this research paper in exactly 40-50 words.

Title: {title}

Abstract: {abstract}

Requirements:
- Focus on the main contribution and methodology
- Be specific about what the paper does
- Use academic writing style
- Avoid phrases like "This paper presents" or "The authors propose"
- Start directly with the key contribution

Summary:""")
        ])
        
        self.chain = summary_prompt | self.llm | StrOutputParser()
    
    def summarize_paper(self, paper: Paper) -> Summary:
        """
        Generate summary for a single paper.
        
        Args:
            paper: Paper object with title and abstract
            
        Returns:
            Summary object
        """
        try:
            summary_text = self.chain.invoke({
                "title": paper.title,
                "abstract": paper.abstract or ""
            })
            
            return Summary(
                paper_id=paper.paper_id,
                text=summary_text.strip(),
                model_name="llama-3.3-70b-versatile"
            )
            
        except Exception as e:
            print(f"⚠️ Summary failed for {paper.title[:50]}: {e}")
            return Summary(
                paper_id=paper.paper_id,
                text=paper.abstract[:200] if paper.abstract else "No summary available.",
                model_name="fallback"
            )
    
    def summarize_papers(
        self, 
        papers: List[Paper],
        show_progress: bool = True
    ) -> List[Summary]:
        """
        Generate summaries for multiple papers.
        
        Args:
            papers: List of Paper objects
            show_progress: Whether to print progress
            
        Returns:
            List of Summary objects
        """
        summaries = []
        total = len(papers)
        
        if show_progress:
            print(f"\n📝 Summarizing {total} papers...")
            print("="*60)
        
        for i, paper in enumerate(papers, 1):
            if show_progress:
                print(f"[{i}/{total}] {paper.title[:50]}...")
            
            summary = self.summarize_paper(paper)
            summaries.append(summary)
            
            if show_progress:
                print(f"        ✅ {len(summary.text)} chars")
        
        if show_progress:
            print("="*60)
            print(f"✅ Generated {len(summaries)} summaries")
        
        return summaries


# === Convenience Functions ===

def summarize_paper(paper: Paper) -> Summary:
    """Summarize a single paper."""
    chain = SummaryChain()
    return chain.summarize_paper(paper)


def summarize_papers(papers: List[Paper]) -> List[Summary]:
    """Summarize multiple papers."""
    chain = SummaryChain()
    return chain.summarize_papers(papers)


# === Test ===

if __name__ == "__main__":
    from src.clients.semantic_scholar import SemanticScholarClient
    
    print("="*60)
    print("SUMMARY CHAIN TEST")
    print("="*60)
    
    # Step 1: Get some papers from Semantic Scholar
    print("\n🔍 Fetching papers from Semantic Scholar...")
    client = SemanticScholarClient()
    papers = client.get_top_papers(
        query="transformer neural network",
        top_n=3,
        year_from=2020
    )
    
    # Step 2: Summarize them
    chain = SummaryChain()
    summaries = chain.summarize_papers(papers)
    
    # Step 3: Display results
    print("\n" + "="*60)
    print("SUMMARIES")
    print("="*60)
    
    for i, (paper, summary) in enumerate(zip(papers, summaries), 1):
        print(f"\n[{i}] {paper.title}")
        print(f"    Authors: {paper.citation_key}")
        print(f"    Summary: {summary.text}")
        print(f"    Words: {len(summary.text.split())}")
