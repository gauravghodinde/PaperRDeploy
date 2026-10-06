"""
PDF Extractor for Target Paper.

Extracts Title, Abstract, and Supplementary/Complementary information
from user-uploaded research paper PDFs.

Uses:
- PyPDF2 for text extraction
- LangChain for intelligent extraction of Title, Abstract, Supplementary
"""

import os
from typing import Optional, Dict
from PyPDF2 import PdfReader
from io import BytesIO

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.utils.config import settings
from src.utils.llm_factory import get_llm


class TargetPaperExtractor:
    """
    Extract information from TARGET paper (user-uploaded PDF).
    
    This is the FIRST step in the pipeline:
    1. User uploads their paper PDF
    2. We extract: Title, Abstract, Supplementary Info
    3. Then use these to generate keywords
    4. Then search for related papers
    """
    
    def __init__(self, max_pages: int = 10):
        """
        Initialize extractor.
        
        Args:
            max_pages: Maximum pages to extract from PDF
        """
        self.max_pages = max_pages
        
        # Initialize LLM using factory (respects LLM_PROVIDER setting)
        self.llm = get_llm(temperature=0.1)
        
        # Create extraction chains
        self._setup_chains()
    
    def _setup_chains(self):
        """Set up LangChain extraction chains."""
        
        # Title extraction prompt
        title_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting titles from research papers."),
            ("human", """Extract ONLY the title from this research paper text.
            
Rules:
- Return ONLY the title, nothing else
- Do not include authors, affiliations, or any other text
- The title is usually at the top of the paper

Text:
{text}""")
        ])
        
        # Abstract extraction prompt
        abstract_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting abstracts from research papers."),
            ("human", """Extract ONLY the abstract from this research paper text.

Rules:
- Return ONLY the abstract content
- Do not include the word "Abstract" itself
- If no abstract is found, return "NO_ABSTRACT_FOUND"

Text:
{text}""")
        ])
        
        # SUPPLEMENTARY info extraction prompt (user's improved version)
        supplementary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting supplementary information from research papers."),
            ("human", """You are given text from a research paper.

Extract the SUPPLEMENTARY INFORMATION.

Definition:
Supplementary information includes material that supports, validates, or enables reproduction of the paper's main claims.

Include:
- Additional experimental or methodological details
- Hyperparameters, training settings, and preprocessing steps
- Extra figures, tables, or quantitative results not central to the main narrative
- Ablation studies, extended evaluations, or additional datasets
- Mathematical proofs or derivations
- Model architecture or implementation details

Exclude:
- High-level summaries
- Motivations, applications, or future work
- General background discussion

Output format:
- Use bullet points
- Group related items under clear headings
- Be faithful to the paper (do not infer missing content)
- If no supplementary info found, return "NO_SUPPLEMENTARY_INFO"

Text:
{text}""")
        ])
        
        # COMPLEMENTARY info extraction prompt (user's improved version)
        complementary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting complementary information from research papers."),
            ("human", """You are given text from a research paper.

Extract the COMPLEMENTARY INFORMATION.

Definition:
Complementary information adds context, interpretation, or practical value but is not essential to validate the core technical claims.

Include:
- Background or contextual explanations
- Real-world use cases or applications
- Intuitive explanations or conceptual insights
- Limitations, ethical considerations, or assumptions
- Failure cases or qualitative observations
- Practical implications or deployment considerations
- Future research directions

Exclude:
- Experimental configurations or hyperparameters
- Proofs or technical derivations
- Core quantitative results

Output format:
- Use bullet points
- Group content under meaningful headings
- Keep explanations concise and non-technical where possible

Text:
{text}""")
        ])
        
        # Create chains
        self.title_chain = title_prompt | self.llm | StrOutputParser()
        self.abstract_chain = abstract_prompt | self.llm | StrOutputParser()
        self.supplementary_chain = supplementary_prompt | self.llm | StrOutputParser()
        self.complementary_chain = complementary_prompt | self.llm | StrOutputParser()
        
        # Chunking settings
        # Llama 3.3 70B has 128K context, but we stay safe with ~20K chars per chunk
        self.max_chunk_chars = 20000
        self.chunk_overlap = 500
    
    def chunk_text(self, text: str, max_chars: int = None) -> list:
        """
        Split text into chunks for LLM processing.
        
        Handles the case where paper text is too long for the LLM context window.
        
        Args:
            text: Full text to chunk
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        max_chars = max_chars or self.max_chunk_chars
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            # Try to break at a paragraph or sentence boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', start, end)
                if para_break > start + max_chars // 2:
                    end = para_break
                else:
                    # Look for sentence break
                    sent_break = text.rfind('. ', start, end)
                    if sent_break > start + max_chars // 2:
                        end = sent_break + 1
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap  # Overlap for context
            
            if start < 0:
                start = 0
        
        print(f"📦 Split text into {len(chunks)} chunks")
        return chunks
    
    def extract_text_from_file(
        self, 
        file_path: str,
        max_pages: Optional[int] = None
    ) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            max_pages: Maximum pages to extract
            
        Returns:
            Extracted text
        """
        max_pages = max_pages or self.max_pages
        
        try:
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                text_parts = []
                
                for i, page in enumerate(reader.pages):
                    if i >= max_pages:
                        break
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
                
                full_text = "\n".join(text_parts)
                print(f"📄 Extracted {len(full_text)} characters from {min(len(reader.pages), max_pages)} pages")
                return full_text
                
        except Exception as e:
            print(f"❌ Failed to extract from {file_path}: {e}")
            return ""
    
    def extract_title(self, text: str) -> str:
        """
        Extract title from paper text using LLM.
        
        Args:
            text: First ~2 pages of paper text
            
        Returns:
            Paper title
        """
        try:
            # Use only first 3000 chars for title (it's at the top)
            title = self.title_chain.invoke({"text": text[:3000]})
            return title.strip()
        except Exception as e:
            print(f"⚠️ Title extraction failed: {e}")
            return ""
    
    def extract_abstract(self, text: str) -> str:
        """
        Extract abstract from paper text using LLM.
        
        Args:
            text: First ~3 pages of paper text
            
        Returns:
            Paper abstract
        """
        try:
            # Use first 6000 chars for abstract
            abstract = self.abstract_chain.invoke({"text": text[:6000]})
            if abstract == "NO_ABSTRACT_FOUND":
                return ""
            return abstract.strip()
        except Exception as e:
            print(f"⚠️ Abstract extraction failed: {e}")
            return ""
    
    def extract_supplementary_info(self, text: str) -> str:
        """
        Extract SUPPLEMENTARY information from END of paper.
        
        Supplementary = Extra materials (appendix, extended data, proofs)
        Usually found at the END of the paper.
        
        Args:
            text: Full paper text
            
        Returns:
            Supplementary information
        """
        try:
            # Use last 8000 chars (supplementary usually at end)
            supplementary = self.supplementary_chain.invoke({"text": text[-8000:]})
            if supplementary == "NO_SUPPLEMENTARY_INFO":
                return ""
            return supplementary.strip()
        except Exception as e:
            print(f"⚠️ Supplementary extraction failed: {e}")
            return ""
    
    def extract_complementary_info(self, text: str) -> str:
        """
        Extract COMPLEMENTARY information from MAIN paper content.
        
        Complementary = Key info from main sections (methodology, contributions, 
        datasets, technical approach) - useful for finding related papers.
        Generated from FIRST 5 pages.
        
        Args:
            text: Paper text (first ~5 pages)
            
        Returns:
            Complementary information
        """
        try:
            # Use first 10000 chars (~5 pages of content)
            complementary = self.complementary_chain.invoke({"text": text[:10000]})
            return complementary.strip()
        except Exception as e:
            print(f"⚠️ Complementary extraction failed: {e}")
            return ""
    
    def extract_all(
        self,
        file_path: str,
        supplementary_override: Optional[str] = None,
        complementary_override: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Extract everything from a target paper PDF.
        
        This is the main method to use!
        
        Extracts:
        - Title
        - Abstract  
        - Supplementary Info (from END of paper - appendix, extended data)
        - Complementary Info (from MAIN content - methodology, contributions)

        Supplementary/complementary can be provided manually; if not provided, they
        are left blank (no automatic extraction). Title and abstract are always
        auto-extracted.
        
        Args:
            file_path: Path to PDF file
            supplementary_override: Optional pre-provided supplementary info
            complementary_override: Optional pre-provided complementary info
            
        Returns:
            Dict with title, abstract, supplementary_info, complementary_info, full_text
        """
        print(f"\n📖 Processing target paper: {file_path}")
        print("="*60)
        
        # Step 1: Extract raw text
        full_text = self.extract_text_from_file(file_path)
        if not full_text:
            return {"error": "Failed to extract text from PDF"}
        
        # Step 2: Extract title
        print("🔍 Extracting title...")
        title = self.extract_title(full_text)
        print(f"   Title: {title[:80]}{'...' if len(title) > 80 else ''}")
        
        # Step 3: Extract abstract
        print("🔍 Extracting abstract...")
        abstract = self.extract_abstract(full_text)
        print(f"   Abstract: {len(abstract)} characters")
        
        # Step 4: Supplementary info (user-provided only)
        if supplementary_override is not None:
            print("ℹ️ Using provided supplementary info.")
            supplementary = supplementary_override.strip()
        else:
            print("⏭️ Skipping supplementary extraction (none provided).")
            supplementary = ""

        # Step 5: Complementary info (user-provided only)
        if complementary_override is not None:
            print("ℹ️ Using provided complementary info.")
            complementary = complementary_override.strip()
        else:
            print("⏭️ Skipping complementary extraction (none provided).")
            complementary = ""
        
        print("="*60)
        print("✅ Extraction complete!")
        
        return {
            "title": title,
            "abstract": abstract,
            "supplementary_info": supplementary,
            "complementary_info": complementary,
            "full_text": full_text,
            "file_path": file_path
        }


# === Convenience Function ===

def extract_target_paper(
    file_path: str,
    supplementary_override: Optional[str] = None,
    complementary_override: Optional[str] = None,
) -> Dict[str, str]:
    """
    Convenience function to extract from target paper.
    
    Args:
        file_path: Path to PDF file
        supplementary_override: Optional pre-provided supplementary info
        complementary_override: Optional pre-provided complementary info
        
    Returns:
        Dict with title, abstract, supplementary_info
    """
    extractor = TargetPaperExtractor()
    return extractor.extract_all(
        file_path,
        supplementary_override=supplementary_override,
        complementary_override=complementary_override,
    )


# === Test ===

if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("TARGET PAPER EXTRACTOR TEST")
    print("="*60)
    
    # Check if user provided a PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Use a sample PDF from the project
        pdf_path = input("Enter path to your target paper PDF: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    # Extract
    extractor = TargetPaperExtractor()
    result = extractor.extract_all(
        pdf_path,
        supplementary_override=None,  # provide a string to override
        complementary_override=None,  # provide a string to override
    )
    
    # Display results
    print("\n" + "="*60)
    print("EXTRACTION RESULTS")
    print("="*60)
    
    print(f"\n📌 TITLE:\n{result.get('title', 'N/A')}")
    print(f"\n📝 ABSTRACT:\n{result.get('abstract', 'N/A')}...")
    
    if result.get('supplementary_info'):
        print(f"\n📋 SUPPLEMENTARY INFO (provided):\n{result['supplementary_info']}...")
    else:
        print("\n📋 SUPPLEMENTARY INFO: Not provided")

    if result.get('complementary_info'):
        print(f"\n🔬 COMPLEMENTARY INFO (provided):\n{result['complementary_info']}...")
    else:
        print("\n🔬 COMPLEMENTARY INFO: Not provided")
