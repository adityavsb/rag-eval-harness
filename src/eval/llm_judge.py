"""Structured LLM-as-a-Judge for evaluating RAG generation fidelity and correctness."""

import os
import re

from pydantic import BaseModel, Field


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

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

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

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key and not api_key.startswith("mock-"):
            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key)
                prompt = f"""You are an objective judge evaluating whether an AI assistant's answer is faithful to the provided context.

Context:
{context}

Question: {query}
Answer: {answer}

Task:
1. Break down the answer into atomic factual claims.
2. Check if each claim is directly supported by the context.
3. Output JSON with:
   - "total_claims": integer
   - "grounded_claims": integer
   - "score": float between 0.0 and 1.0
   - "reasoning": brief explanation
"""
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                import json

                data = json.loads(response.choices[0].message.content or "{}")
                return FaithfulnessEvaluation(**data)
            except (RuntimeError, ValueError, KeyError, TypeError, OSError):
                pass

        # Offline deterministic heuristic fallback
        return self._offline_faithfulness(context, answer)

    def _offline_faithfulness(self, context: str, answer: str) -> FaithfulnessEvaluation:
        """Heuristic word overlap and citation check for offline runs."""
        # Tokenize answer into sentences
        sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 10]
        if not sentences:
            return FaithfulnessEvaluation(
                score=1.0, total_claims=0, grounded_claims=0, reasoning="Empty answer."
            )

        context_lower = context.lower()
        grounded_count = 0

        for sentence in sentences:
            words = [w for w in re.findall(r"\w+", sentence.lower()) if len(w) > 3]
            if not words:
                continue
            overlap = sum(1 for w in words if w in context_lower)
            if overlap / len(words) >= 0.5:
                grounded_count += 1

        score = grounded_count / len(sentences) if sentences else 0.0
        return FaithfulnessEvaluation(
            score=round(score, 3),
            total_claims=len(sentences),
            grounded_claims=grounded_count,
            reasoning=f"Heuristic: {grounded_count}/{len(sentences)} sentence claims supported by context vocabulary.",
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

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key and not api_key.startswith("mock-"):
            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key)
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

Output JSON with:
- "rubric_grade": integer 1-5
- "score": float ((rubric_grade - 1) / 4)
- "reasoning": brief explanation
"""
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                import json

                data = json.loads(response.choices[0].message.content or "{}")
                return CorrectnessEvaluation(**data)
            except (RuntimeError, ValueError, KeyError, TypeError, OSError):
                pass

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
        }
        gt_tokens = {t for t in gt_tokens if t not in stopwords and len(t) > 2}

        if not gt_tokens:
            return CorrectnessEvaluation(
                score=1.0, rubric_grade=5, reasoning="No ground truth tokens."
            )

        overlap = len(gt_tokens.intersection(ans_tokens)) / len(gt_tokens)

        if overlap >= 0.75:
            grade = 5
        elif overlap >= 0.50:
            grade = 4
        elif overlap >= 0.30:
            grade = 3
        elif overlap >= 0.15:
            grade = 2
        else:
            grade = 1

        norm_score = (grade - 1) / 4.0
        return CorrectnessEvaluation(
            score=round(norm_score, 3),
            rubric_grade=grade,
            reasoning=f"Heuristic: {overlap:.1%} ground truth keyword overlap (grade {grade}/5).",
        )
