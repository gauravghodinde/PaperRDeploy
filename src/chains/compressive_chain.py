"""
Compressive Chain.

Creates citation-style compressive summaries from paper summaries.
Output format: "[1] Smith et al. (2023) proposed a novel approach..."
"""

import os
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.models.paper import Paper, Summary, CompressiveSummary
from src.utils.config import settings
from src.utils.llm_factory import get_llm


class CompressiveChain:
    """
    Create compressive summaries with citations.
    
    Takes a paper summary and formats it with proper citation.
    """
    
    def __init__(self):
        """
        Initialize compressive chain.
        """
        # Use config-based LLM provider
        self.llm = get_llm(temperature=0.2)
        
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the compressive summary chain."""
        
        compressive_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at writing literature review citations.
Your task is to create a single-sentence summary that naturally incorporates the citation."""),
            ("human", """Create a compressive summary sentence that incorporates this citation naturally.

Citation: {citation_key}
Original Summary: {summary}

Requirements:
- Write ONE flowing sentence
- Start with the citation naturally (e.g., "Smith et al. (2023) proposed..." or "In their work, Smith et al. (2023) demonstrated...")
- Keep it 30-50 words
- Maintain the key information from the original summary
- Use academic writing style

Compressive summary:""")
        ])
        
        self.chain = compressive_prompt | self.llm | StrOutputParser()
    
    def create_compressive_summary(
        self, 
        paper: Paper, 
        summary: Summary,
        citation_number: int
    ) -> CompressiveSummary:
        """
        Create a compressive summary with citation.
        
        Args:
            paper: Paper object
            summary: Summary object
            citation_number: Citation number (1, 2, 3...)
            
        Returns:
            CompressiveSummary object
        """
        try:
            # Generate compressive text
            compressive_text = self.chain.invoke({
                "citation_key": paper.citation_key,
                "summary": summary.text
            })
            
            return CompressiveSummary(
                paper_id=paper.paper_id,
                citation_key=paper.citation_key,
                summary_text=compressive_text.strip(),
                citation_number=citation_number
            )
            
        except Exception as e:
            print(f"⚠️ Compressive summary failed for {paper.title[:50]}: {e}")
            # Fallback: simple format
            fallback_text = f"{paper.citation_key} {summary.text}"
            return CompressiveSummary(
                paper_id=paper.paper_id,
                citation_key=paper.citation_key,
                summary_text=fallback_text,
                citation_number=citation_number
            )
    
    def create_compressive_summaries(
        self,
        papers: List[Paper],
        summaries: List[Summary],
        show_progress: bool = True
    ) -> List[CompressiveSummary]:
        """
        Create compressive summaries for multiple papers.
        
        Args:
            papers: List of Paper objects
            summaries: List of Summary objects (same order as papers)
            show_progress: Whether to print progress
            
        Returns:
            List of CompressiveSummary objects
        """
        compressive_summaries = []
        total = len(papers)
        
        if show_progress:
            print(f"\n📝 Creating {total} compressive summaries...")
            print("="*60)
        
        for i, (paper, summary) in enumerate(zip(papers, summaries), 1):
            if show_progress:
                print(f"[{i}/{total}] {paper.citation_key}...")
            
            comp_summary = self.create_compressive_summary(
                paper=paper,
                summary=summary,
                citation_number=i
            )
            compressive_summaries.append(comp_summary)
            
            if show_progress:
                print(f"        ✅ [{i}] {len(comp_summary.summary_text)} chars")
        
        if show_progress:
            print("="*60)
            print(f"✅ Created {len(compressive_summaries)} compressive summaries")
        
        return compressive_summaries


# === Convenience Function ===

def create_compressive_summaries(
    papers: List[Paper], 
    summaries: List[Summary]
) -> List[CompressiveSummary]:
    """Create compressive summaries for papers."""
    chain = CompressiveChain()
    return chain.create_compressive_summaries(papers, summaries)


# === Test ===

if __name__ == "__main__":
    from src.clients.semantic_scholar import SemanticScholarClient
    from src.chains.summary_chain import SummaryChain
    
    print("="*60)
    print("COMPRESSIVE CHAIN TEST")
    print("="*60)
    
    # Step 1: Get papers
    print("\n🔍 Fetching papers from Semantic Scholar...")
    ss_client = SemanticScholarClient()
    papers = ss_client.get_top_papers(
        query="deep learning medical imaging",
        top_n=3,
        year_from=2020
    )
    
    # Step 2: Create summaries
    print("\n📝 Creating summaries...")
    summary_chain = SummaryChain()
    summaries = summary_chain.summarize_papers(papers)
    
    # Step 3: Create compressive summaries
    print("\n🔗 Creating compressive summaries...")
    comp_chain = CompressiveChain()
    comp_summaries = comp_chain.create_compressive_summaries(papers, summaries)
    
    # Step 4: Display results
    print("\n" + "="*60)
    print("COMPRESSIVE SUMMARIES")
    print("="*60)
    
    for comp in comp_summaries:
        print(f"\n{comp.formatted}")
        print(f"    ({len(comp.summary_text.split())} words)")
