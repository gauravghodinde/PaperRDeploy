"""
Data models for research papers.
Simplified - only essential fields.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Author(BaseModel):
    """Author information."""
    name: str
    author_id: Optional[str] = None


class Paper(BaseModel):
    """
    Research paper model.
    Only essential fields for the pipeline.
    """
    
    # Core fields (required)
    paper_id: str
    title: str
    abstract: str = ""
    
    # Authors
    authors: List[Author] = Field(default_factory=list)
    
    # Metadata (for citations)
    year: Optional[int] = None
    venue: Optional[str] = None
    citation_count: int = 0
    
    @property
    def formatted_authors(self) -> str:
        """Get formatted author string for citation."""
        if not self.authors:
            return "Unknown Author"
        
        names = [a.name for a in self.authors if a.name]
        
        if len(names) == 0:
            return "Unknown Author"
        elif len(names) == 1:
            return names[0]
        elif len(names) == 2:
            return f"{names[0]} and {names[1]}"
        else:
            # Get last name of first author
            first_name = names[0]
            last_name = first_name.split()[-1] if first_name else "Unknown"
            return f"{last_name} et al."
    
    @property
    def citation_key(self) -> str:
        """Get citation key like 'Smith et al. (2023)'."""
        year_str = str(self.year) if self.year else "n.d."
        return f"{self.formatted_authors} ({year_str})"


class TargetPaper(BaseModel):
    """
    Target paper (user's uploaded PDF).
    Separate from Paper because it has different fields.
    """
    title: str
    abstract: str
    supplementary_info: str = ""
    complementary_info: str = ""
    file_path: str = ""


class Summary(BaseModel):
    """Summary of a paper."""
    paper_id: str
    text: str
    model_name: str = "unknown"
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())


class CompressiveSummary(BaseModel):
    """Compressive summary with citation information."""
    paper_id: str
    citation_key: str
    summary_text: str
    citation_number: Optional[int] = None
    
    @property
    def formatted(self) -> str:
        """Get formatted summary with citation."""
        if self.citation_number:
            return f"[{self.citation_number}] {self.summary_text}"
        return self.summary_text
