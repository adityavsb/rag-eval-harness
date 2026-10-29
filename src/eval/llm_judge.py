"""Structured LLM-as-a-Judge for evaluating RAG generation fidelity and correctness."""

import json
import os
import re
from typing import Any

from pydantic import BaseModel, Field


def extract_json_payload(response_text: str) -> dict[str, Any]:
    """Extract valid JSON from LLM response, stripping reasoning tags and code fences."""
    cleaned = response_text.strip()

    # 1. Strip reasoning tags (<think>...</think> or [REASONING]...[/REASONING])
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"\[REASONING\].*?\[/REASONING\]", "", cleaned, flags=re.DOTALL).strip()

    # 2. Extract the LAST markdown code fence if multiple exist
    fence_matches = list(re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE))
    if fence_matches:
        cleaned = fence_matches[-1].group(1).strip()

    # 3. Find outermost braces
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx : end_idx + 1]

    # 4. Parse with trailing comma recovery
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        sanitized = re.sub(r",\s*([\}\]])", r"\1", cleaned)
        try:
            return json.loads(sanitized)
        except Exception:
            import ast

            return ast.literal_eval(cleaned)


def normalize_decision_dict(raw_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize decision payloads from perplexity decider or standard JSON schemas."""
    d = dict(raw_dict)
    # 1. Unwrap envelopes
    for env_key in ["decision", "verdict_details", "result", "output", "data"]:
        if env_key in d and isinstance(d[env_key], dict):
            inner = d[env_key]
            for k, v in inner.items():
                if k not in d or d[k] is None:
                    d[k] = v

    # 2. Extract confidence / score / probability
    if "score" not in d:
        if "probability" in d:
            d["score"] = float(d["probability"])
        elif "confidence" in d:
            d["score"] = float(d["confidence"])
        elif "rubric_grade" in d:
            d["score"] = max(0.0, min(1.0, (int(d["rubric_grade"]) - 1) / 4.0))
        elif "noul" in d:
            d["score"] = 1.0 if d["noul"] in (True, 1, "yes", "true") else 0.0

    return d


class FaithfulnessEvaluation(BaseModel):
    """Result of evaluating whether generated claims are strictly grounded in context."""

    score: float = Field(
        ge=0.0, le=1.0, description="1.0 means fully grounded; 0.0 means ungrounded/hallucinated."
    )
    total_claims: int = 0
    grounded_claims: int = 0
    reasoning: str = ""


class CorrectnessEvaluation(BaseModel):
    """Result of evaluating generated answer against golden reference answer."""

    score: float = Field(ge=0.0, le=1.0, description="Normalized correctness score (0.0 to 1.0).")
    rubric_grade: int = Field(ge=1, le=5, description="1 to 5 rubric score.")
    reasoning: str = ""


class LLMJudge:
    """Evaluates generation quality using structured rubrics with offline heuristic fallbacks."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        mode: str = "offline",
        provider: str = "openai",
    ):
        self.model_name = model_name
        self.mode = mode
        self.provider = provider

    @property
    def is_decision_model(self) -> bool:
        """Returns True if model uses OpenRouter /api/alpha/decisions endpoint."""
        target = self.model_name.lower()
        return "decider" in target or "decision" in target or "pplx-decider" in target

    def evaluate_sample(
        self, query: str, context: str, answer: str, ground_truth: str
    ) -> tuple[FaithfulnessEvaluation, CorrectnessEvaluation, dict[str, Any]]:
        """Evaluates both faithfulness and correctness, batching questions into a single decisions call if supported."""
        # Check for refusal
        if "insufficient information" in ground_truth.lower() and "insufficient information" in answer.lower():
            return (
                FaithfulnessEvaluation(
                    score=1.0,
                    total_claims=1,
                    grounded_claims=1,
                    reasoning="Model appropriately identified insufficient context and refused.",
                ),
                CorrectnessEvaluation(
                    score=1.0,
                    rubric_grade=5,
                    reasoning="Correctly refused out-of-scope query matching ground truth.",
                ),
                {"usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}},
            )
        if "insufficient information" in ground_truth.lower() and "insufficient information" not in answer.lower():
            return (
                FaithfulnessEvaluation(
                    score=0.0,
                    total_claims=1,
                    grounded_claims=0,
                    reasoning="Failed to refuse: answered an out-of-scope adversarial query.",
                ),
                CorrectnessEvaluation(
                    score=0.0,
                    rubric_grade=1,
                    reasoning="Failed to refuse: answered an out-of-scope adversarial query.",
                ),
                {"usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}},
            )

        if self.mode == "live" and self.is_decision_model:
            if not os.environ.get("OPENROUTER_API_KEY"):
                raise ValueError(
                    f"Live mode evaluation requested for LLMJudge with model '{self.model_name}', "
                    "but OPENROUTER_API_KEY environment variable is missing."
                )

            from src.llm.provider import call_openrouter_decisions

            state = {
                "context": context,
                "query": query,
                "answer": answer,
                "ground_truth": ground_truth,
            }
            questions = {
                "faithfulness": {
                    "type": "noul",
                    "instructions": "Is the generated answer strictly supported by the context without foreign entities or hallucinations?",
                    "criteria": {
                        "true": "All factual claims in the answer are directly grounded in the context.",
                        "false": "The answer contains ungrounded assertions or foreign numerical data.",
                    },
                },
                "correctness": {
                    "type": "noul",
                    "instructions": "Does the generated answer accurately match the ground truth reference answer?",
                    "criteria": {
                        "true": "The answer covers core factual details consistent with the ground truth.",
                        "false": "The answer contradicts or misses critical facts from the ground truth.",
                    },
                },
            }
            res = call_openrouter_decisions(
                model=self.model_name,
                state=state,
                questions=questions,
            )
            answers = res.get("answers", {})
            f_noul = float(answers.get("faithfulness", {}).get("noul", 0.0))
            c_noul = float(answers.get("correctness", {}).get("noul", 0.0))

            f_score = round(max(0.0, min(1.0, f_noul)), 4)
            c_score = round(max(0.0, min(1.0, c_noul)), 4)
            grade = max(1, min(5, int(1 + round(c_score * 4))))

            faith_eval = FaithfulnessEvaluation(
                score=f_score,
                total_claims=1,
                grounded_claims=1 if f_score >= 0.5 else 0,
                reasoning=f"Perplexity Decider live evaluation: faithfulness noul={f_score:.4f}",
            )
            corr_eval = CorrectnessEvaluation(
                score=c_score,
                rubric_grade=grade,
                reasoning=f"Perplexity Decider live evaluation: correctness noul={c_score:.4f} (grade {grade}/5)",
            )
            usage = res.get("usage", {})
            return faith_eval, corr_eval, usage

        # Otherwise evaluate individually
        f_eval = self.evaluate_faithfulness(query, context, answer)
        c_eval = self.evaluate_correctness(query, answer, ground_truth)
        return f_eval, c_eval, {}

    def evaluate_faithfulness(
        self, query: str, context: str, answer: str
    ) -> FaithfulnessEvaluation:
        """Evaluates whether statements in the answer are strictly supported by the retrieved context."""
        # Check for refusal
        if "insufficient information" in answer.lower():
            return FaithfulnessEvaluation(
                score=1.0,
                total_claims=1,
                grounded_claims=1,
                reasoning="Model appropriately identified insufficient context and refused to hallucinate.",
            )

        if self.mode == "live":
            provider = self.provider.lower()
            model_target = self.model_name
            if (
                provider == "openrouter"
                or "openrouter" in model_target
                or "deepseek" in model_target
                or "qwen" in model_target
                or "perplexity" in model_target
                or "pplx-decider" in model_target
                or ":free" in model_target
                or "nemotron" in model_target
            ):
                if not os.environ.get("OPENROUTER_API_KEY"):
                    raise ValueError(
                        f"Live mode evaluation requested for LLMJudge with model '{self.model_name}', "
                        "but OPENROUTER_API_KEY environment variable is missing."
                    )
            elif provider == "openai" or "gpt" in model_target:
                if not os.environ.get("OPENAI_API_KEY"):
                    raise ValueError(
                        f"Live mode evaluation requested for LLMJudge with model '{self.model_name}', "
                        "but OPENAI_API_KEY environment variable is missing."
                    )
            elif provider == "anthropic" or "claude" in model_target:
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    raise ValueError(
                        f"Live mode evaluation requested for LLMJudge with model '{self.model_name}', "
                        "but ANTHROPIC_API_KEY environment variable is missing."
                    )

            if self.is_decision_model:
                from src.llm.provider import call_openrouter_decisions

                state = {
                    "context": context,
                    "query": query,
                    "answer": answer,
                }
                questions = {
                    "faithfulness": {
                        "type": "noul",
                        "instructions": "Is the generated answer strictly supported by the context without foreign entities or hallucinations?",
                        "criteria": {
                            "true": "All factual claims in the answer are directly grounded in the context.",
                            "false": "The answer contains ungrounded assertions or foreign numerical data.",
                        },
                    }
                }
                res = call_openrouter_decisions(
                    model=self.model_name,
                    state=state,
                    questions=questions,
                )
                f_ans = res.get("answers", {}).get("faithfulness", {})
                score = float(f_ans.get("noul", 0.0))
                return FaithfulnessEvaluation(
                    score=round(max(0.0, min(1.0, score)), 4),
                    total_claims=1,
                    grounded_claims=1 if score >= 0.5 else 0,
                    reasoning=f"Evaluated via OpenRouter decisions ({self.model_name}): noul={score:.4f}",
                )

            from src.llm.provider import call_llm

            prompt = f"""You are an objective judge evaluating whether an AI assistant's answer is faithful to the provided context.

Context:
{context}

Question: {query}
Answer: {answer}

Task:
1. Break down the answer into atomic factual claims.
2. Check if each claim is directly supported by the context.
3. Output ONLY a valid JSON object matching:
{{
  "total_claims": 1,
  "grounded_claims": 1,
  "score": 1.0,
  "reasoning": "brief explanation"
}}
"""
            llm_res = call_llm(model=self.model_name, prompt=prompt, temperature=0.0)
            raw_dict = extract_json_payload(llm_res.content)
            norm_dict = normalize_decision_dict(raw_dict)
            total = int(norm_dict.get("total_claims", 1))
            score = float(
                norm_dict.get(
                    "score",
                    1.0 if norm_dict.get("is_discrepancy") is False else 0.0,
                )
            )
            grounded = int(norm_dict.get("grounded_claims", round(score * total)))
            return FaithfulnessEvaluation(
                score=max(0.0, min(1.0, score)),
                total_claims=total,
                grounded_claims=grounded,
                reasoning=str(norm_dict.get("reasoning", "Evaluated by judge model.")),
            )

        # Offline deterministic heuristic fallback
        return self._offline_faithfulness(context, answer)

    def _offline_faithfulness(self, context: str, answer: str) -> FaithfulnessEvaluation:
        """Heuristic word overlap and entity consistency check for offline runs."""
        # Tokenize answer into sentences
        sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 10]
        if not sentences:
            return FaithfulnessEvaluation(
                score=1.0, total_claims=0, grounded_claims=0, reasoning="Empty answer."
            )

        context_lower = context.lower()
        context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", context_lower))
        grounded_count = 0

        for sentence in sentences:
            words = [w for w in re.findall(r"\w+", sentence.lower()) if len(w) > 3]
            if not words:
                continue

            # Number consistency check: if sentence introduces foreign numbers, it is ungrounded
            sentence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", sentence.lower()))
            if sentence_numbers and not sentence_numbers.issubset(context_numbers):
                continue

            overlap = sum(1 for w in words if w in context_lower)
            if overlap / len(words) >= 0.55:
                grounded_count += 1

        score = grounded_count / len(sentences) if sentences else 0.0
        return FaithfulnessEvaluation(
            score=round(score, 3),
            total_claims=len(sentences),
            grounded_claims=grounded_count,
            reasoning=f"Heuristic: {grounded_count}/{len(sentences)} sentence claims supported by context vocabulary and numeric entities.",
        )

    def evaluate_correctness(
        self, query: str, answer: str, ground_truth: str
    ) -> CorrectnessEvaluation:
        """Evaluates generated answer against the ground-truth reference answer."""
        # Both refused
        if "insufficient information" in ground_truth.lower():
            if "insufficient information" in answer.lower():
                return CorrectnessEvaluation(
                    score=1.0,
                    rubric_grade=5,
                    reasoning="Correctly refused out-of-scope query matching ground truth.",
                )
            else:
                return CorrectnessEvaluation(
                    score=0.0,
                    rubric_grade=1,
                    reasoning="Failed to refuse: answered an out-of-scope adversarial query.",
                )

        if self.mode == "live":
            provider = self.provider.lower()
            model_target = self.model_name
            if (
                provider == "openrouter"
                or "openrouter" in model_target
                or "deepseek" in model_target
                or "qwen" in model_target
                or "perplexity" in model_target
                or "pplx-decider" in model_target
                or ":free" in model_target
                or "nemotron" in model_target
            ):
                if not os.environ.get("OPENROUTER_API_KEY"):
                    raise ValueError(
                        f"Live mode evaluation requested for LLMJudge with model '{self.model_name}', "
                        "but OPENROUTER_API_KEY environment variable is missing."
                    )
            elif provider == "openai" or "gpt" in model_target:
                if not os.environ.get("OPENAI_API_KEY"):
                    raise ValueError(
                        f"Live mode evaluation requested for LLMJudge with model '{self.model_name}', "
                        "but OPENAI_API_KEY environment variable is missing."
                    )
            elif provider == "anthropic" or "claude" in model_target:
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    raise ValueError(
                        f"Live mode evaluation requested for LLMJudge with model '{self.model_name}', "
                        "but ANTHROPIC_API_KEY environment variable is missing."
                    )

            if self.is_decision_model:
                from src.llm.provider import call_openrouter_decisions

                state = {
                    "query": query,
                    "answer": answer,
                    "ground_truth": ground_truth,
                }
                questions = {
                    "correctness": {
                        "type": "noul",
                        "instructions": "Does the generated answer accurately match the ground truth reference answer?",
                        "criteria": {
                            "true": "The answer covers core factual details consistent with the ground truth.",
                            "false": "The answer contradicts or misses critical facts from the ground truth.",
                        },
                    }
                }
                res = call_openrouter_decisions(
                    model=self.model_name,
                    state=state,
                    questions=questions,
                )
                c_ans = res.get("answers", {}).get("correctness", {})
                score = float(c_ans.get("noul", 0.0))
                rubric_grade = max(1, min(5, int(1 + round(score * 4))))
                return CorrectnessEvaluation(
                    score=round(max(0.0, min(1.0, score)), 4),
                    rubric_grade=rubric_grade,
                    reasoning=f"Evaluated via OpenRouter decisions ({self.model_name}): noul={score:.4f}",
                )

            from src.llm.provider import call_llm

            prompt = f"""You are an objective judge evaluating answer correctness against ground truth.

Question: {query}
Ground Truth: {ground_truth}
Generated Answer: {answer}

Rubric:
5 - Fully correct: covers all core facts accurately.
4 - Mostly correct: covers main facts with minor omissions.
3 - Partially correct: has some correct facts but misses key information.
2 - Mostly incorrect: factual contradictions or largely missing.
1 - Completely incorrect or ungrounded.

Output ONLY a valid JSON object matching:
{{
  "rubric_grade": 5,
  "score": 1.0,
  "reasoning": "brief explanation"
}}
"""
            llm_res = call_llm(model=self.model_name, prompt=prompt, temperature=0.0)
            raw_dict = extract_json_payload(llm_res.content)
            norm_dict = normalize_decision_dict(raw_dict)
            if "score" in norm_dict and "rubric_grade" not in norm_dict:
                score = float(norm_dict["score"])
                grade = round(1 + score * 4)
            elif "rubric_grade" in norm_dict:
                grade = int(norm_dict["rubric_grade"])
                score = float(norm_dict.get("score", (grade - 1) / 4.0))
            else:
                grade = 5 if norm_dict.get("is_discrepancy") is False else 1
                score = (grade - 1) / 4.0

            return CorrectnessEvaluation(
                score=max(0.0, min(1.0, score)),
                rubric_grade=max(1, min(5, grade)),
                reasoning=str(norm_dict.get("reasoning", "Evaluated by judge model.")),
            )

        # Offline deterministic lexical & semantic overlap fallback
        return self._offline_correctness(answer, ground_truth)

    def _offline_correctness(self, answer: str, ground_truth: str) -> CorrectnessEvaluation:
        """Heuristic n-gram token overlap scoring against ground truth."""
        ans_tokens = set(re.findall(r"\w+", answer.lower()))
        gt_tokens = set(re.findall(r"\w+", ground_truth.lower()))

        # Remove common stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "and",
            "or",
            "to",
            "in",
            "for",
            "with",
            "of",
            "on",
            "at",
            "by",
            "from",
            "be",
            "this",
            "that",
            "it",
            "all",
            "must",
            "used",
            "exclusively",
        }
        gt_tokens = {t for t in gt_tokens if t not in stopwords and len(t) > 2}

        if not gt_tokens:
            return CorrectnessEvaluation(
                score=1.0, rubric_grade=5, reasoning="No ground truth tokens."
            )

        # Check numeric discrepancies between answer and ground truth
        ans_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", answer.lower()))
        gt_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", ground_truth.lower()))
        num_mismatch = bool(ans_nums and gt_nums and not ans_nums.intersection(gt_nums))

        overlap = len(gt_tokens.intersection(ans_tokens)) / len(gt_tokens)

        if num_mismatch and overlap < 0.8:
            grade = 1 if overlap < 0.3 else 2
        elif overlap >= 0.60:
            grade = 5
        elif overlap >= 0.40:
            grade = 4
        elif overlap >= 0.25:
            grade = 3
        elif overlap >= 0.12:
            grade = 2
        else:
            grade = 1

        norm_score = (grade - 1) / 4.0
        return CorrectnessEvaluation(
            score=round(norm_score, 3),
            rubric_grade=grade,
            reasoning=f"Heuristic: {overlap:.1%} ground truth keyword overlap (grade {grade}/5).",
        )
