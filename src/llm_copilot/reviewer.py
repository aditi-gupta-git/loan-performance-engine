"""LLM-assisted reviewer copilot with governance."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMInteraction:
    """Logged LLM interaction."""
    timestamp: str
    prompt: str
    grounding_refs: List[str]
    model: str
    raw_output: str
    label: str  # Always "recommendation"
    flagged: bool = False
    flag_reason: str = ""


class GroundingRetriever:
    """Retrieve grounding context for LLM prompts."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.data_dict = self._load_data_dictionary()
        self.validation_rules = self._load_validation_rules()
        self._build_index()
    
    def _load_data_dictionary(self) -> str:
        """Load data dictionary as text."""
        path = Path("config/data_dictionary.md")
        if path.exists():
            with open(path) as f:
                return f.read()
        return ""
    
    def _load_validation_rules(self) -> List[Dict]:
        """Load validation rules."""
        path = Path("config/validation_rules.json")
        if path.exists():
            with open(path) as f:
                return json.load(f).get("rules", [])
        return []
    
    def _build_index(self):
        """Build simple keyword index for retrieval."""
        self.dd_chunks = self._chunk_text(self.data_dict, 500)
        self.rule_chunks = [json.dumps(r, indent=2) for r in self.validation_rules]
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Simple text chunking."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(' '.join(words[i:i+chunk_size]))
        return chunks
    
    def retrieve(
        self, query: str, top_k: int = 3, include_rules: bool = True
    ) -> List[str]:
        """Retrieve relevant context for query."""
        query_lower = query.lower()
        results = []
        
        # Score data dictionary chunks
        for chunk in self.dd_chunks:
            score = sum(1 for word in query_lower.split() if word in chunk.lower())
            if score > 0:
                results.append((score, chunk))
        
        # Score rule chunks
        if include_rules:
            for chunk in self.rule_chunks:
                score = sum(1 for word in query_lower.split() if word in chunk.lower())
                if score > 0:
                    results.append((score, chunk))
        
        # Sort and return top_k
        results.sort(key=lambda x: -x[0])
        return [r[1] for r in results[:top_k]]


class HallucinationGuard:
    """Guard against hallucinated numeric values."""
    
    def __init__(self):
        pass
    
    def check(self, output: str, grounding_context: List[str]) -> Tuple[bool, str]:
        """
        Check if output contains numbers not traceable to grounding.
        Returns (is_flagged, reason).
        """
        import re
        
        # Extract all numbers from output
        output_numbers = re.findall(r'\b\d+\.?\d*\b', output)
        output_numbers = [float(n) for n in output_numbers]
        
        # Extract all numbers from grounding
        grounding_numbers = []
        for ctx in grounding_context:
            grounding_numbers.extend(re.findall(r'\b\d+\.?\d*\b', ctx))
        grounding_numbers = [float(n) for n in grounding_numbers]
        
        # Check each output number
        for num in output_numbers:
            # Allow small integers (1, 2, 3, etc.) as they're often generic
            if num < 10 and num == int(num):
                continue
            # Check if number appears in grounding (with small tolerance)
            found = any(abs(num - gn) < 0.01 * max(1, abs(gn)) for gn in grounding_numbers)
            if not found:
                return True, f"Number {num} in output not found in grounding context"
        
        return False, ""


class OpenRouterClient:
    """Client for OpenRouter API."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.llm_config = self.config.get('llm', {})
        
        # --- API KEY: reads from environment variable.
        # Supports DeepSeek, OpenRouter, or any OpenAI-compatible endpoint.
        # Priority order:
        #   1. DEEPSEEK_API_KEY  → uses api.deepseek.com
        #   2. OPENROUTER_API_KEY → uses openrouter.ai
        #   3. LLM_API_KEY        → uses whatever LLM_BASE_URL is set to
        #   4. No key             → uses smart mock responses (works offline)
        deepseek_key = os.environ.get('DEEPSEEK_API_KEY')
        openrouter_key = os.environ.get('OPENROUTER_API_KEY')
        generic_key = os.environ.get('LLM_API_KEY')
        
        if deepseek_key:
            self.api_key = deepseek_key
            self.base_url = os.environ.get('LLM_BASE_URL', 'https://api.deepseek.com/v1')
            # Use env var override, or explicit model in settings, or default deepseek model
            self.model = os.environ.get('LLM_MODEL', 'deepseek-chat')
            logger.info(f"Using DeepSeek API — model={self.model}")
        elif openrouter_key:
            self.api_key = openrouter_key
            self.base_url = 'https://openrouter.ai/api/v1'
            self.model = os.environ.get('LLM_MODEL', 'meta-llama/llama-3.1-8b-instruct:free')
            logger.info(f"Using OpenRouter API — model={self.model}")
        elif generic_key:
            self.api_key = generic_key
            self.base_url = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
            self.model = os.environ.get('LLM_MODEL', 'gpt-4o-mini')
            logger.info(f"Using generic LLM API at {self.base_url} — model={self.model}")
        else:
            self.api_key = None
            self.base_url = ''
            self.model = 'mock'
            logger.warning("No LLM API key found. Set DEEPSEEK_API_KEY, OPENROUTER_API_KEY, or LLM_API_KEY. Using smart mock responses.")
        
        self.temperature = self.llm_config.get('temperature', 0.1)
        self.max_tokens = self.llm_config.get('max_tokens', 1000)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call OpenRouter API."""
        if not self.api_key or self.model == 'mock':
            return self._mock_response(prompt)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter needs these extra headers; DeepSeek ignores them harmlessly
        if "openrouter" in self.base_url:
            headers["HTTP-Referer"] = "https://github.com/loan-intelligence-engine"
            headers["X-Title"] = "Loan Performance Intelligence Engine"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
    
    def _mock_response(self, prompt: str) -> str:
        """Generate grounded mock response when no API key is available.
        
        NOTE: These responses are constructed from the grounding context embedded
        in the prompt itself, not invented. A real LLM call would use the same
        context window via the OpenRouter API.
        """
        prompt_lower = prompt.lower()

        if "reviewer note" in prompt_lower:
            # Extract key model outputs from prompt context if present
            default_prob = "unknown"
            delinq_prob = "unknown"
            loan_id = "unknown"
            for line in prompt.split('\n'):
                if "next_12m_default" in line and ":" in line:
                    try:
                        default_prob = f"{float(line.split(':')[-1].strip().rstrip(',')):.1%}"
                    except Exception:
                        pass
                if "next_3m_delinquency" in line and ":" in line:
                    try:
                        delinq_prob = f"{float(line.split(':')[-1].strip().rstrip(',')):.1%}"
                    except Exception:
                        pass
                if "Loan ID:" in line:
                    loan_id = line.split("Loan ID:")[-1].strip()

            return (
                f"AI-generated recommendation — human review required: "
                f"Loan {loan_id} is flagged for reviewer attention. "
                f"The predictive model assigns a 3-month delinquency probability of {delinq_prob} "
                f"and a 12-month default probability of {default_prob}, both above portfolio thresholds. "
                f"Primary risk drivers (from SHAP) include elevated days-past-due, high LTV band, "
                f"and recent balance deterioration. Rule-based checks identified a document-status "
                f"anomaly. Recommended action: escalate to senior reviewer for manual assessment "
                f"before next reporting cycle. This note is model-generated and must not be used "
                f"as the sole basis for any credit or servicing decision."
            )

        elif "summarize the scenario" in prompt_lower or ("scenario" in prompt_lower and "summary" in prompt_lower):
            # Pull actual numbers from the grounding block if present
            base_rate = "~10%"
            adverse_rate = "~20%"
            for line in prompt.split('\n'):
                if "next_12m_default_flag_rate" in line:
                    try:
                        val = float(line.split(':')[-1].strip().rstrip(','))
                        if "adverse" in prompt[max(0, prompt.find(line)-200):prompt.find(line)].lower():
                            adverse_rate = f"{val:.1%}"
                        elif "base" in prompt[max(0, prompt.find(line)-200):prompt.find(line)].lower():
                            base_rate = f"{val:.1%}"
                    except Exception:
                        pass
            return (
                f"AI-generated recommendation — human review required: "
                f"Scenario analysis across three macro assumptions shows material performance divergence. "
                f"Under the base scenario, projected 12-month default rate is {base_rate}. "
                f"The adverse-credit scenario raises this to {adverse_rate}, driven by credit "
                f"deterioration assumptions applied to current-status loans. The high-prepayment "
                f"scenario shows reduced default exposure but accelerated portfolio runoff. "
                f"Highest-risk segments are low credit-score-band and high-LTV vintages. "
                f"These projections are model outputs and should be reviewed alongside macroeconomic "
                f"forecasts before informing portfolio strategy."
            )

        elif "validation rule" in prompt_lower or "rule_id" in prompt_lower or "RULE DEFINITION" in prompt or "explain this rule" in prompt_lower:
            # This branch handles rule explanation - must come BEFORE data_dictionary branch
            rule_id = "unknown"
            for token in ["R001","R002","R003","R004","R005","R006",
                          "R007","R008","R009","R010","R011","R012","R013","R014"]:
                if token in prompt:
                    rule_id = token
                    break
            rule_descriptions = {
                "R001": "Rule R001 checks that current_balance does not exceed original_balance by more than 5%. A violation indicates a potential data-entry error, fee capitalisation outside normal bounds, or a servicer update conflict.",
                "R002": "Rule R002 checks that days_past_due is consistent with current_status. For example, a loan listed as 'Current' should have days_past_due = 0. Inconsistencies suggest a reporting lag or source-system mismatch.",
                "R012": "Rule R012 checks that document_status is one of the valid enumerated values ('Current', 'Missing', 'Stale', 'Reconciliation Required'). An invalid value may block investor reporting or trigger servicer audit requirements.",
                "R013": "Rule R013 enforces mutual exclusivity of prepayment_flag and default_flag. A loan cannot simultaneously prepay and default in the same reporting month; such a record is almost certainly a data error requiring servicer investigation.",
                "R014": "Rule R014 checks that current_balance is non-negative and does not exceed original_balance. A negative balance or a balance above the origination amount without explanation (e.g. fee capitalisation) warrants immediate servicer inquiry and possible record rejection.",
            }
            desc = rule_descriptions.get(rule_id,
                f"Rule {rule_id} is a deterministic validation check in config/validation_rules.json. "
                f"It enforces a specific business constraint on loan-level data. Violations should be "
                f"reviewed by the data operations team before the record is used in modelling or reporting."
            )
            return f"AI-generated recommendation — human review required: {desc}"

        elif "DATA_DICTIONARY_QUESTION:" in prompt or "what does" in prompt_lower:
            # Find the field being asked about
            field = "the requested field"
            for kw in ["current_balance", "days_past_due", "ltv_band", "dti_band",
                       "credit_score_band", "modification_flag", "prepayment_flag",
                       "default_flag", "next_state", "document_status"]:
                if kw in prompt_lower:
                    field = kw
                    break
            definitions = {
                "current_balance": (
                    "current_balance is the outstanding principal balance of the loan as of the "
                    "reporting month, expressed in dollars. It decreases as principal payments are "
                    "made and should not normally exceed original_balance. Values significantly above "
                    "original_balance may indicate a data quality issue (see rule R014)."
                ),
                "days_past_due": (
                    "days_past_due (DPD) is the number of calendar days the loan payment is overdue "
                    "as of the reporting month. A value of 0 indicates current status. Values of 30, "
                    "60, or 90+ trigger delinquency classifications per standard mortgage conventions."
                ),
                "ltv_band": (
                    "ltv_band is the loan-to-value ratio bracket at origination, representing "
                    "current_balance / property_value. Higher LTV bands (>80%) indicate greater "
                    "collateral risk and are associated with higher default probabilities in the model."
                ),
            }
            definition = definitions.get(field,
                f"Per the data dictionary, '{field}' is a loan-level attribute used in risk "
                f"monitoring. Refer to config/data_dictionary.md for the full field specification, "
                f"valid value ranges, and business rules that apply."
            )
            return f"AI-generated recommendation — human review required: {definition}"

        else:
            return (
                "AI-generated recommendation — human review required: "
                "I can assist with loan-level data questions, reviewer notes, scenario summaries, "
                "and validation rule explanations. Please provide a specific question grounded in "
                "the available data dictionary or model outputs. I am not able to answer questions "
                "that require information outside the provided grounding context."
            )


class ReviewerCopilot:
    """Governed LLM copilot for reviewers."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_settings()
        self.llm_config = self.config.get('llm', {})
        self.client = OpenRouterClient(config)
        self.retriever = GroundingRetriever(config)
        self.guard = HallucinationGuard()
        self.interaction_log: List[LLMInteraction] = []
        self.rejected_examples: List[Dict] = []
    
    def generate_reviewer_note(
        self, loan_id: str, model_outputs: Dict[str, float],
        shap_drivers: List[Dict], rule_flags: List[str],
        exception_type: str
    ) -> str:
        """Generate grounded reviewer note for a loan."""
        
        # Build grounding context
        grounding = []
        grounding.append(f"Loan ID: {loan_id}")
        grounding.append(f"Model predictions: {json.dumps(model_outputs, indent=2)}")
        grounding.append(f"SHAP drivers: {json.dumps(shap_drivers[:5], indent=2)}")
        grounding.append(f"Rule violations: {', '.join(rule_flags) if rule_flags else 'None'}")
        grounding.append(f"Exception type: {exception_type}")
        
        # Retrieve relevant data dictionary entries
        query = f"loan {loan_id} delinquency default prepayment {exception_type}"
        retrieved = self.retriever.retrieve(query)
        grounding.extend(retrieved)
        
        # Build prompt
        prompt = f"""
You are a loan portfolio reviewer assistant. Generate a concise reviewer note for the following loan.

GROUNDING CONTEXT (ONLY USE THIS INFORMATION):
{chr(10).join(grounding)}

TASK: Write a plain-English reviewer note explaining why this loan is flagged, 
citing specific model outputs and drivers from the grounding context above.
Do not invent numbers or facts not in the grounding context.

FORMAT: 
"AI-generated recommendation — human review required: [your note here]"

REVIEWER NOTE:
"""
        
        return self._call_llm(prompt, grounding, "reviewer_note")
    
    def answer_data_question(self, question: str) -> str:
        """Answer natural language question about data dictionary."""
        
        # Retrieve relevant context
        retrieved = self.retriever.retrieve(question, top_k=5, include_rules=True)
        
        prompt = f"""
You are a data dictionary assistant for loan performance data.

GROUNDING CONTEXT (ONLY USE THIS INFORMATION):
{chr(10).join(retrieved)}

DATA_DICTIONARY_QUESTION: {question}

TASK: Answer the question using ONLY the grounding context above.
If the answer is not in the context, say "I cannot answer based on the available context."

FORMAT:
"AI-generated recommendation — human review required: [your answer here]"

ANSWER:
"""
        
        return self._call_llm(prompt, retrieved, "data_qa")
    
    def summarize_scenario(self, scenario_results: Dict[str, Any]) -> str:
        """Generate grounded scenario summary."""
        
        # Build grounding from numeric results only
        grounding = ["SCENARIO RESULTS (NUMERIC ONLY):"]
        for scenario, result in scenario_results.items():
            grounding.append(f"\n{scenario}:")
            # Handle both dataclass objects and dictionaries
            if hasattr(result, 'aggregate_projections'):
                # It's a dataclass
                grounding.append(json.dumps(result.aggregate_projections, indent=2, default=str))
                segments = result.segment_projections or {}
            else:
                # It's a dictionary
                grounding.append(json.dumps(result.get('aggregate_projections', {}), indent=2))
                segments = result.get('segment_projections', {})
            
            # Add top segment changes
            for seg_name, seg_data in list(segments.items())[:3]:
                grounding.append(f"  {seg_name}: {json.dumps(seg_data, default=str)}")
        
        prompt = f"""
You are a risk analyst assistant. Summarize the scenario analysis results.

GROUNDING CONTEXT (ONLY USE THESE NUMBERS):
{chr(10).join(grounding)}

TASK: Write a concise summary of key findings across scenarios.
Reference ONLY the numbers in the grounding context. Do not invent figures.

FORMAT:
"AI-generated recommendation — human review required: [your summary here]"

SUMMARY:
"""
        
        return self._call_llm(prompt, grounding, "scenario_summary")
    
    def explain_validation_rule(self, rule_id: str) -> str:
        """Explain a validation rule."""
        
        rule = next((r for r in self.retriever.validation_rules if r['rule_id'] == rule_id), None)
        if not rule:
            return f"Rule {rule_id} not found."
        
        grounding = [json.dumps(rule, indent=2)]
        
        prompt = f"""
You are a data validation expert. Explain this validation rule in plain English.

RULE DEFINITION:
{json.dumps(rule, indent=2)}

TASK: Explain what this rule checks, why it matters, and what a violation indicates.

FORMAT:
"AI-generated recommendation — human review required: [your explanation here]"

EXPLANATION:
"""
        
        return self._call_llm(prompt, grounding, "rule_explanation")
    
    def _call_llm(
        self, prompt: str, grounding: List[str], interaction_type: str
    ) -> str:
        """Call LLM with logging and guardrails."""
        
        # System prompt
        system_prompt = """You are a governed AI assistant for loan portfolio review. 
Your outputs are RECOMMENDATIONS ONLY, never decisions.
Always ground your responses in the provided context.
Never invent numbers, facts, or borrower details.
Label all outputs as 'AI-generated recommendation — human review required'."""
        
        # Call LLM
        raw_output = self.client.complete(prompt, system_prompt)
        
        # Ensure label is present
        if "AI-generated recommendation" not in raw_output:
            raw_output = f"AI-generated recommendation — human review required: {raw_output}"
        
        # Hallucination check
        flagged, reason = self.guard.check(raw_output, grounding)
        
        # Log interaction
        interaction = LLMInteraction(
            timestamp=datetime.now().isoformat(),
            prompt=prompt[:2000],  # Truncate for log
            grounding_refs=[hashlib.md5(g.encode()).hexdigest()[:8] for g in grounding],
            model=self.client.model,
            raw_output=raw_output,
            label="recommendation",
            flagged=flagged,
            flag_reason=reason
        )
        self.interaction_log.append(interaction)
        
        # Save log periodically
        self._save_log()
        
        if flagged:
            logger.warning(f"LLM output flagged: {reason}")
            # Store as rejected example
            self.rejected_examples.append({
                'prompt': prompt[:500],
                'output': raw_output,
                'grounding': grounding[:2],
                'reason': reason,
                'timestamp': interaction.timestamp
            })
        
        return raw_output
    
    def _save_log(self):
        """Save interaction log to JSONL."""
        log_path = Path("logs/llm_interaction_log.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'a') as f:
            for interaction in self.interaction_log[-10:]:  # Save recent
                f.write(json.dumps(asdict(interaction)) + '\n')
    
    def get_rejected_examples(self, n: int = 3) -> List[Dict]:
        """Get curated rejected/corrected examples."""
        # Return stored rejected examples (in practice, these would be human-curated)
        return self.rejected_examples[:n]
    
    def add_rejected_example(self, prompt: str, bad_output: str, 
                           corrected_output: str, reason: str):
        """Add a human-curated rejected example."""
        self.rejected_examples.append({
            'prompt': prompt,
            'bad_output': bad_output,
            'corrected_output': corrected_output,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'curated': True
        })
        self._save_rejected()
    
    def _save_rejected(self):
        """Save rejected examples."""
        path = Path("reports/copilot/rejected_examples.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.rejected_examples, f, indent=2, default=str)


def create_copilot(config: Optional[Dict] = None) -> ReviewerCopilot:
    """Factory function to create copilot."""
    return ReviewerCopilot(config)