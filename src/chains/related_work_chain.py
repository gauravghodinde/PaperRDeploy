"""
Related Work Chain.

Generates the final Related Work section from compressive summaries.
Produces a cohesive, well-structured literature review.
"""

import os
from typing import List, Optional, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.models.paper import Paper, CompressiveSummary
from src.utils.config import settings
from src.utils.llm_factory import get_llm


class RelatedWorkChain:
    """
    Generate a complete Related Work section using multiple models.
    
    Takes compressive summaries and creates a cohesive literature review
    using Gemini, Llama, OpenAI, and DeepSeek models.
    """
    
    def __init__(self):
        """
        Initialize related work chain with all four models.
        """
        # Initialize all four models
        self.gemini_llm = get_llm(temperature=0.4, provider="gemini")
        self.llama_llm = get_llm(temperature=0.4, provider="groq")  # Default Groq uses Llama
        self.openai_llm = self._get_openai_oss_llm(temperature=0.4)  # OpenAI/GPT-OSS-120b via Groq
        self.deepseek_llm = self._get_deepseek_llm(temperature=0.4)  # DeepSeek
        
        self._setup_chains()
    
    def _get_openai_oss_llm(self, temperature: float = 0.4):
        """Get OpenAI/GPT-OSS-120b LLM instance via Groq API."""
        try:
            from langchain_groq import ChatGroq
            import os
            
            os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
            return ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=temperature
            )
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI/GPT-OSS-120b: {e}")
            return None
    
    def _get_deepseek_llm(self, temperature: float = 0.4):
        """Get DeepSeek LLM instance via OpenAI-compatible API."""
        try:
            return get_llm(temperature=temperature, provider="deepseek")
        except Exception as e:
            print(f"⚠️ Failed to initialize DeepSeek: {e}")
            return None
    
    def _setup_chains(self):
        """Set up the related work generation chains for all models."""
        
        related_work_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert academic writer specializing in literature reviews.
Your task is to write a cohesive Related Work section that:
- Must be EXACTLY 300 words (count carefully)
- Groups related papers by theme
- Uses smooth transitions between ideas
- Maintains academic writing style
- Properly references all cited works"""),
            ("human", """Write a Related Work section for a research paper.

Target Paper Context:
Title: {target_title}
Abstract: {target_abstract}

Papers to cite (with their compressive summaries):
{compressive_summaries}

Requirements:
1. Write EXACTLY 300 words (not 299, not 301 - count every word)
2. Write 2-4 cohesive paragraphs
3. Group papers by theme/topic where appropriate
4. Use proper citation format [1], [2], etc.
5. Connect papers to the target paper's research area
6. Use transitions like "Furthermore,", "In contrast,", "Similarly,"
7. Highlight key contributions of each cited work
8. End by positioning how these works relate to the target paper

CRITICAL: The output must be EXACTLY 300 words. Count carefully.

Related Work:""")
        ])
        
        self.gemini_chain = related_work_prompt | self.gemini_llm | StrOutputParser()
        self.llama_chain = related_work_prompt | self.llama_llm | StrOutputParser()
        
        if self.openai_llm:
            self.openai_chain = related_work_prompt | self.openai_llm | StrOutputParser()
        else:
            self.openai_chain = None
        
        if self.deepseek_llm:
            self.deepseek_chain = related_work_prompt | self.deepseek_llm | StrOutputParser()
        else:
            self.deepseek_chain = None
        
        # === Ensemble Prompt (uses Gemini to synthesize all outputs) ===
        ensemble_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert academic editor specializing in synthesizing multiple versions of literature reviews.

Your task is to create the BEST possible Related Work section by:
- Analyzing multiple AI-generated versions
- Selecting the strongest elements from each
- Ensuring coherent flow and academic style
- Maintaining proper citation format [1], [2], etc.
- Producing EXACTLY 300 words"""),
            ("human", """Create the BEST Related Work section by synthesizing these multiple versions:

Target Paper:
Title: {target_title}
Abstract: {target_abstract}

--- VERSION 1 (Gemini) ---
{gemini_output}

--- VERSION 2 (LLaMA) ---
{llama_output}

--- VERSION 3 (GPT) ---
{openai_output}

--- VERSION 4 (DeepSeek) ---
{deepseek_output}

---

Instructions:
1. Analyze the strengths of each version (structure, flow, coverage, academic tone)
2. Synthesize the BEST elements into a single cohesive Related Work section
3. Ensure all citations [1]-[15] are properly integrated
4. Write EXACTLY 300 words
5. Maintain smooth transitions and thematic grouping
6. Position the target paper's contribution at the end

Generate the synthesized Related Work section:""")
        ])
        
        self.ensemble_chain = ensemble_prompt | self.gemini_llm | StrOutputParser()
    
    def generate_related_work_all_models(
        self,
        target_title: str,
        target_abstract: str,
        compressive_summaries: List[CompressiveSummary]
    ) -> Dict[str, str]:
        """
        Generate the Related Work section using all three models.
        
        Args:
            target_title: Title of the target paper
            target_abstract: Abstract of the target paper
            compressive_summaries: List of CompressiveSummary objects
            
        Returns:
            Dict with keys 'gemini', 'llama', 'openai' containing generated text
        """
        # Format compressive summaries for the prompt
        formatted_summaries = []
        for comp in compressive_summaries:
            formatted_summaries.append(f"[{comp.citation_number}] {comp.summary_text}")
        
        summaries_text = "\n".join(formatted_summaries)
        
        print(f"\n📝 Generating Related Work section with all three models...")
        print(f"   Target: {target_title[:60]}...")
        print(f"   Citing: {len(compressive_summaries)} papers")
        print("="*60)
        
        results = {}
        
        # Generate with Gemini
        try:
            print("\n🔵 Generating with Gemini...")
            gemini_result = self.gemini_chain.invoke({
                "target_title": target_title,
                "target_abstract": target_abstract,
                "compressive_summaries": summaries_text
            })
            results["gemini"] = gemini_result.strip()
            print("✅ Gemini Related Work generated!")
        except Exception as e:
            print(f"⚠️ Gemini generation failed: {e}")
            results["gemini"] = None
        
        # Generate with Llama
        try:
            print("\n🟠 Generating with Llama-3.3-70b-versatile...")
            llama_result = self.llama_chain.invoke({
                "target_title": target_title,
                "target_abstract": target_abstract,
                "compressive_summaries": summaries_text
            })
            results["llama"] = llama_result.strip()
            print("✅ Llama Related Work generated!")
        except Exception as e:
            print(f"⚠️ Llama generation failed: {e}")
            results["llama"] = None
        
        # Generate with OpenAI
        if self.openai_chain:
            try:
                print("\n⚪ Generating with OpenAI/GPT-OSS-120b (via Groq)...")
                openai_result = self.openai_chain.invoke({
                    "target_title": target_title,
                    "target_abstract": target_abstract,
                    "compressive_summaries": summaries_text
                })
                results["openai"] = openai_result.strip()
                print("✅ OpenAI/GPT-OSS-120b Related Work generated!")
            except Exception as e:
                print(f"⚠️ OpenAI/GPT-OSS-120b generation failed: {e}")
                results["openai"] = None
        else:
            results["openai"] = None
        
        # Generate with DeepSeek
        if self.deepseek_chain:
            try:
                print("\n🔷 Generating with DeepSeek...")
                deepseek_result = self.deepseek_chain.invoke({
                    "target_title": target_title,
                    "target_abstract": target_abstract,
                    "compressive_summaries": summaries_text
                })
                results["deepseek"] = deepseek_result.strip()
                print("✅ DeepSeek Related Work generated!")
            except Exception as e:
                print(f"⚠️ DeepSeek generation failed: {e}")
                results["deepseek"] = None
        else:
            results["deepseek"] = None
        
        return results
    
    def generate_ensemble(
        self,
        target_title: str,
        target_abstract: str,
        model_results: Dict[str, str]
    ) -> str:
        """
        Generate an ensemble Related Work by synthesizing outputs from all models.
        
        Uses Gemini to analyze all model outputs and create the best combined version.
        
        Args:
            target_title: Title of the target paper
            target_abstract: Abstract of the target paper
            model_results: Dict with outputs from each model
            
        Returns:
            Synthesized Related Work section
        """
        print("\n" + "="*60)
        print("🌟 ENSEMBLE GENERATION (Synthesizing all models with Gemini)")
        print("="*60)
        
        # Get outputs from each model (use empty string if not available)
        gemini_output = model_results.get("gemini", "") or "[Not available]"
        llama_output = model_results.get("llama", "") or "[Not available]"
        openai_output = model_results.get("openai", "") or "[Not available]"
        deepseek_output = model_results.get("deepseek", "") or "[Not available]"
        
        # Count available outputs
        available_count = sum(1 for v in [model_results.get("gemini"), 
                                          model_results.get("llama"), 
                                          model_results.get("openai"), 
                                          model_results.get("deepseek")] if v)
        
        print(f"   Synthesizing from {available_count} model outputs...")
        
        if available_count == 0:
            print("⚠️ No model outputs available for ensemble!")
            return None
        
        try:
            ensemble_result = self.ensemble_chain.invoke({
                "target_title": target_title,
                "target_abstract": target_abstract,
                "gemini_output": gemini_output,
                "llama_output": llama_output,
                "openai_output": openai_output,
                "deepseek_output": deepseek_output
            })
            
            print("✅ Ensemble Related Work generated!")
            return ensemble_result.strip()
            
        except Exception as e:
            print(f"⚠️ Ensemble generation failed: {e}")
            # Fallback: return the first available output
            for model in ["gemini", "llama", "openai", "deepseek"]:
                if model_results.get(model):
                    print(f"   Falling back to {model} output")
                    return model_results[model]
            return None
    
    def generate_related_work(
        self,
        target_title: str,
        target_abstract: str,
        compressive_summaries: List[CompressiveSummary]
    ) -> str:
        """
        Generate the Related Work section (using default/first available model).
        
        Args:
            target_title: Title of the target paper
            target_abstract: Abstract of the target paper
            compressive_summaries: List of CompressiveSummary objects
            
        Returns:
            Generated Related Work section as string
        """
        # Format compressive summaries for the prompt
        formatted_summaries = []
        for comp in compressive_summaries:
            formatted_summaries.append(f"[{comp.citation_number}] {comp.summary_text}")
        
        summaries_text = "\n".join(formatted_summaries)
        
        print(f"\n📝 Generating Related Work section...")
        print(f"   Target: {target_title[:60]}...")
        print(f"   Citing: {len(compressive_summaries)} papers")
        print("="*60)
        
        try:
            related_work = self.llama_chain.invoke({
                "target_title": target_title,
                "target_abstract": target_abstract,
                "compressive_summaries": summaries_text
            })
            
            print("✅ Related Work generated!")
            return related_work.strip()
            
        except Exception as e:
            print(f"⚠️ Related Work generation failed: {e}")
            # Fallback: simple concatenation
            fallback = "Related Work\n\n"
            for comp in compressive_summaries:
                fallback += f"{comp.formatted} "
            return fallback.strip()
    
    def generate_with_references(
        self,
        target_title: str,
        target_abstract: str,
        compressive_summaries: List[CompressiveSummary],
        papers: List[Paper]
    ) -> dict:
        """
        Generate Related Work with formatted references using all three models.
        
        Args:
            target_title: Title of the target paper
            target_abstract: Abstract of the target paper
            compressive_summaries: List of CompressiveSummary objects
            papers: List of Paper objects (for reference list)
            
        Returns:
            Dict with results from all three models:
            {
                'gemini': {...},
                'llama': {...},
                'openai': {...}
            }
            Each containing 'related_work' and 'references' sections
        """
        # Generate related work using all three models
        model_results = self.generate_related_work_all_models(
            target_title=target_title,
            target_abstract=target_abstract,
            compressive_summaries=compressive_summaries
        )
        
        # Generate references list (same for all models)
        references = []
        for i, paper in enumerate(papers, 1):
            ref = f"[{i}] {paper.formatted_authors}. \"{paper.title}\""
            if paper.venue:
                ref += f", {paper.venue}"
            if paper.year:
                ref += f", {paper.year}"
            ref += "."
            references.append(ref)
        
        references_text = "\n".join(references)
        
        # Return results from all models
        results = {}
        for model_name, related_work in model_results.items():
            if related_work:
                results[model_name] = {
                    "related_work": related_work,
                    "references": references_text,
                    "paper_count": len(papers)
                }
            else:
                results[model_name] = {
                    "related_work": None,
                    "references": references_text,
                    "paper_count": len(papers)
                }
        
        # Generate ensemble output (synthesized from all models)
        ensemble_output = self.generate_ensemble(
            target_title=target_title,
            target_abstract=target_abstract,
            model_results=model_results
        )
        
        results["ensemble"] = {
            "related_work": ensemble_output,
            "references": references_text,
            "paper_count": len(papers)
        }
        
        return results


# === Convenience Function ===

def generate_related_work(
    target_title: str,
    target_abstract: str,
    compressive_summaries: List[CompressiveSummary]
) -> str:
    """Generate related work section."""
    chain = RelatedWorkChain()
    return chain.generate_related_work(
        target_title, target_abstract, compressive_summaries
    )


# === Test ===

if __name__ == "__main__":
    from src.clients.semantic_scholar import SemanticScholarClient
    from src.chains.summary_chain import SummaryChain
    from src.chains.compressive_chain import CompressiveChain
    
    print("="*60)
    print("RELATED WORK CHAIN TEST")
    print("="*60)
    
    # Simulate a target paper
    target_title = "Deep Learning for Medical Image Segmentation"
    target_abstract = """We propose a novel deep learning architecture for 
    medical image segmentation that combines transformer attention mechanisms 
    with convolutional neural networks. Our approach achieves state-of-the-art 
    results on multiple medical imaging benchmarks."""
    
    # Step 1: Get related papers
    print("\n🔍 Fetching related papers...")
    ss_client = SemanticScholarClient()
    papers = ss_client.get_top_papers(
        query="deep learning medical image segmentation",
        top_n=5,
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
    
    # Step 4: Generate related work
    rw_chain = RelatedWorkChain()
    result = rw_chain.generate_with_references(
        target_title=target_title,
        target_abstract=target_abstract,
        compressive_summaries=comp_summaries,
        papers=papers
    )
    
    # Display results
    print("\n" + "="*60)
    print("GENERATED RELATED WORK")
    print("="*60)
    print(result["related_work"])
    
    print("\n" + "="*60)
    print("REFERENCES")
    print("="*60)
    print(result["references"])
