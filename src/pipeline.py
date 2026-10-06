"""
Related Work Generator Pipeline.

Main orchestrator that connects all components to generate
a complete Related Work section from a target paper PDF.

Stages
------
1. READ    : Extract title, abstract, supp/comp info from the PDF (TargetPaperExtractor)
2. SEE     : Classify each information unit as SUPPLEMENTARY or COMPLEMENTARY (SeeStage)
3. SYNTHESIZE: Multi-LLM related work generation (RelatedWorkChain)
"""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict

# Import all components
from src.clients.pdf_extractor import TargetPaperExtractor
from src.clients.keyword_extractor import KeywordExtractor
from src.clients.semantic_scholar import SemanticScholarClient
from src.chains.summary_chain import SummaryChain
from src.chains.compressive_chain import CompressiveChain
from src.chains.related_work_chain import RelatedWorkChain
from src.models.paper import Paper, Summary, CompressiveSummary

# Stage 2: See
try:
    from src.see_stage import SeeStage
    _SEE_STAGE_AVAILABLE = True
except ImportError:
    _SEE_STAGE_AVAILABLE = False


class RelatedWorkPipeline:
    """
    Main pipeline for generating Related Work sections.
    
    Flow:
    1. Extract info from target paper PDF
    2. Generate keywords from extracted content
    3. Search Semantic Scholar for related papers
    4. Summarize each related paper
    5. Create compressive summaries with citations
    6. Generate cohesive Related Work section
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        
        self._log("🚀 Initializing Related Work Pipeline (Read–See–Synthesize)...")
        
        # Stage 1: Read
        self.pdf_extractor      = TargetPaperExtractor()
        self.keyword_extractor  = KeywordExtractor()
        self.ss_client          = SemanticScholarClient()

        # Stage 2: See (ML-based supp/comp classifier)
        if _SEE_STAGE_AVAILABLE:
            self._log("🔍 Initializing See Stage classifier...")
            self.see_stage = SeeStage(auto_load=True)
            if self.see_stage.is_ready:
                self._log("   ✅ See Stage classifier ready")
            else:
                self._log("   ⚠️  See Stage running in pass-through mode (no saved model)")
        else:
            self.see_stage = None
            self._log("   ⚠️  See Stage not available (missing details_classifier module)")

        # Stage 3: Synthesize
        self.summary_chain      = SummaryChain()
        self.compressive_chain  = CompressiveChain()
        self.related_work_chain = RelatedWorkChain()
        
        self._log("✅ All components initialized!")
    
    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def run(
        self,
        pdf_path: str,
        num_papers: int = 15,
        year_from: Optional[int] = None,
        save_output: bool = True,
        output_dir: str = "data/outputs",
        supplementary_override: Optional[str] = None,
        complementary_override: Optional[str] = None
    ) -> Dict:
        """
        Run the complete pipeline.
        
        Args:
            pdf_path: Path to target paper PDF
            num_papers: Number of related papers to find (default: 15)
            year_from: Only include papers from this year onwards
            save_output: Whether to save results to file
            output_dir: Directory for output files
            supplementary_override: Optional pre-provided supplementary info
            complementary_override: Optional pre-provided complementary info
            
        Returns:
            Dict with all pipeline outputs
        """
        self._log("\n" + "="*70)
        self._log("🔬 RELATED WORK GENERATOR PIPELINE")
        self._log("="*70)
        
        start_time = datetime.now()
        
        # ============================================================
        # STAGE 1 (READ): Extract from Target Paper PDF
        # ============================================================
        self._log("\n📖 STAGE 1 (READ): Extracting from target paper...")
        
        paper_info = self.pdf_extractor.extract_all(
            pdf_path,
            supplementary_override=supplementary_override,
            complementary_override=complementary_override
        )
        
        if "error" in paper_info:
            return {"error": paper_info["error"]}
        
        self._log(f"   ✅ Title: {paper_info['title'][:60]}...")
        self._log(f"   ✅ Abstract: {len(paper_info['abstract'])} chars")
        self._log(f"   ✅ Supplementary: {len(paper_info.get('supplementary_info', '') or '')} chars")
        self._log(f"   ✅ Complementary: {len(paper_info.get('complementary_info', '') or '')} chars")

        # ============================================================
        # STAGE 2 (SEE): Classify graphical-abstract information units
        # ============================================================
        self._log("\n👁️  STAGE 2 (SEE): Classifying graphical abstract evidence...")

        see_output = None
        if self.see_stage is not None:
            see_output = self.see_stage.process(
                abstract          = paper_info["abstract"],
                supplementary_raw = paper_info.get("supplementary_info", "") or "",
                complementary_raw = paper_info.get("complementary_info", "") or "",
            )
            self._log(f"   ✅ Classified {see_output.classified_units}/{see_output.total_units} units")
            self._log(f"   ✅ Supplementary: {len(see_output.supplementary)} units")
            self._log(f"   ✅ Complementary: {len(see_output.complementary)} units")
            self._log(f"   ✅ Model used: {see_output.model_used}")
            # Replace raw text with ML-classified structured text
            paper_info["supplementary_info"] = see_output.supplementary_text or paper_info.get("supplementary_info", "")
            paper_info["complementary_info"] = see_output.complementary_text or paper_info.get("complementary_info", "")
        else:
            self._log("   ⚠️  See Stage unavailable; using LLM-extracted text as-is")
        
        # ============================================================
        # STAGE 3a (SYNTHESIZE – Keywords): Generate search keywords
        # ============================================================
        self._log("\n🔑 STAGE 3a (SYNTHESIZE): Generating keywords...")
        
        # Use title + abstract; include supplementary if provided
        keywords = self.keyword_extractor.extract_keywords(
            title=paper_info['title'],
            abstract=paper_info['abstract'],
            supplementary=paper_info.get('supplementary_info'),
            use_llm=True  # LLM refines keywords
        )
        
        self._log(f"   ✅ Generated {len(keywords)} refined keywords:")
        for i, kw in enumerate(keywords, 1):
            self._log(f"      [{i}] {kw}")
        
        # ============================================================
        # STAGE 3b (SYNTHESIZE – Retrieval): Search Semantic Scholar
        # ============================================================
        self._log(f"\n🔍 STAGE 3b (SYNTHESIZE): Searching for {num_papers} related papers...")
        
        # Use top keywords as search query
        search_query = " ".join(keywords[:5])
        self._log(f"   Query: '{search_query}'")
        
        papers = self.ss_client.get_top_papers(
            query=search_query,
            top_n=num_papers,
            year_from=year_from
        )
        
        self._log(f"   ✅ Found {len(papers)} related papers")
        
        # ============================================================
        # STAGE 3c (SYNTHESIZE – Summaries): Summarize related papers
        # ============================================================
        self._log(f"\n📝 STAGE 3c (SYNTHESIZE): Summarizing {len(papers)} papers...")
        
        summaries = self.summary_chain.summarize_papers(
            papers,
            show_progress=self.verbose
        )
        
        # ============================================================
        # STAGE 3d (SYNTHESIZE – Compression): Compressive summaries
        # ============================================================
        self._log(f"\n🔗 STAGE 3d (SYNTHESIZE): Creating compressive summaries...")
        
        comp_summaries = self.compressive_chain.create_compressive_summaries(
            papers=papers,
            summaries=summaries,
            show_progress=self.verbose
        )
        
        # ============================================================
        # STAGE 3e (SYNTHESIZE – Generation): Multi-LLM generation
        # ============================================================
        self._log(f"\n📄 STAGE 3e (SYNTHESIZE): Generating Related Work section...")
        
        result = self.related_work_chain.generate_with_references(
            target_title=paper_info['title'],
            target_abstract=paper_info['abstract'],
            compressive_summaries=comp_summaries,
            papers=papers
        )
        
        # ============================================================
        # Compile Final Output
        # ============================================================
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile See Stage stats if available
        see_stats = None
        if see_output:
            see_stats = {
                "total_units":         see_output.total_units,
                "classified_units":    see_output.classified_units,
                "supplementary_units": len(see_output.supplementary),
                "complementary_units": len(see_output.complementary),
                "classification_rate": see_output.classification_rate,
                "model_used":          see_output.model_used,
            }

        # Create output with all model results
        output = {
            "target_paper": {
                "title":              paper_info['title'],
                "abstract":           paper_info['abstract'],
                "supplementary_info": paper_info.get('supplementary_info', ''),
                "complementary_info": paper_info.get('complementary_info', ''),
                "pdf_path":           pdf_path
            },
            "see_stage":  see_stats,    # Stage 2 classification statistics
            "keywords":   keywords,
            "related_papers_count": len(papers),
            "related_papers": [
                {
                    "title":     p.title,
                    "authors":   p.citation_key,
                    "year":      p.year,
                    "citations": p.citation_count
                }
                for p in papers
            ],
            "model_results": result,  # Contains results from all LLM models
            "metadata": {
                "generated_at":      datetime.now().isoformat(),
                "duration_seconds":  duration,
                "num_papers_cited":  len(papers),
                "pipeline_version":  "Read-See-Synthesize v2.0",
            }
        }
        
        # Save output if requested
        if save_output:
            self._save_output(output, output_dir, paper_info['title'])
        
        # Print final summary
        self._log("\n" + "="*70)
        self._log("✅ PIPELINE COMPLETE!")
        self._log("="*70)
        self._log(f"   Duration: {duration:.1f} seconds")
        self._log(f"   Papers cited: {len(papers)}")
        
        # Count generated models
        generated_models = sum(1 for model in result.values() if model and model.get("related_work"))
        self._log(f"   Models used: {generated_models}/3")
        
        return output
    
    def _save_output(self, output: Dict, output_dir: str, title: str):
        """Save output to files - creates 3 text files for each model."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create safe filename from title
        safe_title = "".join(c for c in title[:50] if c.isalnum() or c in " -_").strip()
        safe_title = safe_title.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = os.path.join(output_dir, f"{safe_title}_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=str, ensure_ascii=False)
        self._log(f"\n💾 Saved JSON: {json_path}")
        
        # Save 3 separate text files for each model
        model_names = {
            "gemini": "Gemini-2.5-Flash-Lite",
            "llama": "Llama-3.3-70b-Versatile",
            "openai": "OpenAI-GPT-OSS-120b (via Groq)"
        }
        
        model_results = output.get("model_results", {})
        
        for model_key, model_display_name in model_names.items():
            model_data = model_results.get(model_key)
            
            if model_data and model_data.get("related_work"):
                txt_path = os.path.join(
                    output_dir, 
                    f"{safe_title}_{timestamp}_related_work_{model_key}.txt"
                )
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"RELATED WORK - {model_display_name}\n")
                    f.write("="*60 + "\n\n")
                    f.write(model_data["related_work"])
                    f.write("\n\n")
                    f.write("REFERENCES\n")
                    f.write("="*60 + "\n\n")
                    f.write(model_data["references"])
                
                self._log(f"💾 Saved TXT ({model_key}): {txt_path}")
            else:
                self._log(f"⚠️ Skipped {model_key}: No output generated")


# === Convenience Function ===

def generate_related_work(
    pdf_path: str, 
    num_papers: int = 15,
    supplementary_override: Optional[str] = None,
    complementary_override: Optional[str] = None
) -> Dict:
    """
    Generate related work from a PDF.
    
    Args:
        pdf_path: Path to target paper PDF
        num_papers: Number of related papers to find
        supplementary_override: Optional pre-provided supplementary info
        complementary_override: Optional pre-provided complementary info
        
    Returns:
        Dict with related_work and references
    """
    pipeline = RelatedWorkPipeline()
    return pipeline.run(
        pdf_path, 
        num_papers=num_papers,
        supplementary_override=supplementary_override,
        complementary_override=complementary_override
    )


# === CLI Entry Point ===

if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("🔬 RELATED WORK GENERATOR")
    print("="*70)
    
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("\n📄 Enter path to your paper PDF: ").strip()
    
    # Validate
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)
    
    # Get number of papers (optional)
    num_papers = 15
    if len(sys.argv) > 2:
        try:
            num_papers = int(sys.argv[2])
        except ValueError:
            pass
    
    # Run pipeline
    pipeline = RelatedWorkPipeline()
    result = pipeline.run(
        pdf_path=pdf_path,
        num_papers=num_papers,
        save_output=True
    )
    
    # Display output
    print("\n" + "="*70)
    print("📄 GENERATED RELATED WORK")
    print("="*70)
    
    model_results = result.get("model_results", {})
    
    # Display results from all models
    for model_name in ["gemini", "llama", "openai"]:
        model_data = model_results.get(model_name)
        if model_data and model_data.get("related_work"):
            print(f"\n🔵 {model_name.upper()}")
            print("-" * 70)
            print(model_data["related_work"])
        else:
            print(f"\n⚠️ {model_name.upper()}: No output available")
    
    print("\n" + "="*70)
    print("📚 REFERENCES (shared across all models)")
    print("="*70)
    
    # Get references from any available model
    for model_data in model_results.values():
        if model_data and model_data.get("references"):
            print(model_data["references"])
            break
