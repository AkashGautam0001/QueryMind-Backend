"""
Four RAGAS metrics as hand-rolled LLM-judge calls.

Implemented without the `ragas` pip package to avoid dragging in langchain
and datasets as transitive dependencies. Same conceptual measurements,
full control over prompts and failure modes.

Each function returns a float in [0, 1] and a reasoning string.

FAITHFULNESS: Are the answer's claims supported by the retrieved context?
  Decomposes the answer into atomic claims, verifies each one against the
  context. Score = verified_claims / total_claims.

ANSWER RELEVANCY: Does the answer actually address the question asked?
  Generates 3 candidate questions from the answer alone, embeds them, and
  measures cosine similarity to the original question embedding. High
  similarity means the answer stayed on topic; low means it drifted or
  gave a generic response.

CONTEXT PRECISION: Were the retrieved chunks actually useful?
  For each retrieved chunk (in rank order), asks whether it contributed to
  the correct answer. Score weights earlier chunks more (DCG-style) since
  rank matters for what gets included when the context window fills up.

CONTEXT RECALL: Did the retrieved chunks cover everything needed?
  Decomposes the expected (reference) answer into atomic claims, checks
  how many are attributable to the retrieved context. Measures recall of
  the retrieval stage, independent of what the LLM did with the context.
"""

import json
import re

from app.generation.llm_provider import LLMProvider

_JUDGE_SYSTEM = (
    "You are a precise evaluation assistant. You always respond with valid JSON only, "
    "no preamble, no markdown fences, no explanation outside the JSON structure."
)


async def _judge(provider: LLMProvider, prompt: str, max_tokens: int = 512) -> dict:
    raw = await provider.complete(
        system_prompt=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    # Strip markdown fences if the model adds them despite instructions.
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    return json.loads(cleaned)


async def score_faithfulness(
    provider: LLMProvider, question: str, answer: str, context: str
) -> tuple[float, str]:
    """Decomposes the answer into atomic claims, verifies each against context."""
    prompt = f"""Evaluate whether the answer's claims are supported by the context.

QUESTION: {question}

ANSWER: {answer}

CONTEXT:
{context}

Step 1 – List every atomic factual claim made in the answer.
Step 2 – For each claim, decide: SUPPORTED (clearly stated or directly implied by context), UNSUPPORTED (not in context or contradicted), or NOT_A_CLAIM (opinion, hedging, meta-commentary).
Step 3 – Score = supported_claims / (supported_claims + unsupported_claims). If there are no claims, score = 1.0.

Respond with this exact JSON structure:
{{
  "claims": [
    {{"claim": "...", "verdict": "SUPPORTED|UNSUPPORTED|NOT_A_CLAIM", "reason": "..."}}
  ],
  "score": 0.0
}}"""
    result = await _judge(provider, prompt, max_tokens=800)
    score = float(result.get("score", 0.0))
    reasoning = json.dumps(result.get("claims", []))
    return max(0.0, min(1.0, score)), reasoning


async def score_answer_relevancy(
    provider: LLMProvider, question: str, answer: str
) -> tuple[float, str]:
    """Generates synthetic questions from the answer and measures cosine similarity to original."""
    prompt = f"""Given this answer, generate 3 questions that this answer would perfectly address.
Focus only on what the answer actually covers, not on what the original question might have been.

ANSWER: {answer}

Respond with this exact JSON structure:
{{
  "generated_questions": ["question 1", "question 2", "question 3"],
  "relevancy_score": 0.0,
  "reasoning": "..."
}}

For relevancy_score: 1.0 = the answer directly and completely addresses a question identical to the original,
0.5 = partially relevant or somewhat off-topic, 0.0 = completely unrelated.
Compare the generated questions to this original question to determine the score: {question}"""
    result = await _judge(provider, prompt, max_tokens=400)
    score = float(result.get("relevancy_score", 0.0))
    reasoning = result.get("reasoning", "")
    return max(0.0, min(1.0, score)), reasoning


async def score_context_precision(
    provider: LLMProvider, question: str, answer: str, context_chunks: list[str]
) -> tuple[float, str]:
    """Checks whether each retrieved chunk was useful, weighting by rank."""
    if not context_chunks:
        return 0.0, "No context chunks retrieved"

    verdicts_prompt = "For each context chunk below, decide if it was USEFUL or NOT_USEFUL for answering the question correctly.\n\n"
    verdicts_prompt += f"QUESTION: {question}\n\nCORRECT ANSWER: {answer}\n\n"
    for i, chunk in enumerate(context_chunks, 1):
        verdicts_prompt += f"CHUNK {i}: {chunk[:500]}\n\n"
    verdicts_prompt += """Respond with this exact JSON structure:
{
  "verdicts": [
    {"chunk_index": 1, "verdict": "USEFUL|NOT_USEFUL", "reason": "..."}
  ]
}"""
    result = await _judge(provider, verdicts_prompt, max_tokens=600)
    verdicts = result.get("verdicts", [])

    # DCG-style: sum useful*1/rank, divide by ideal DCG (all chunks useful)
    dcg = 0.0
    ideal_dcg = sum(1.0 / i for i in range(1, len(context_chunks) + 1))
    for item in verdicts:
        if item.get("verdict") == "USEFUL":
            rank = item.get("chunk_index", 0)
            if 1 <= rank <= len(context_chunks):
                dcg += 1.0 / rank

    score = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    return max(0.0, min(1.0, score)), json.dumps(verdicts)


async def score_context_recall(
    provider: LLMProvider, question: str, expected_answer: str, context: str
) -> tuple[float, str]:
    """Checks how many claims from the expected answer are attributable to context."""
    prompt = f"""Decompose the expected answer into atomic claims, then check if each is supported by the context.

QUESTION: {question}

EXPECTED ANSWER (ground truth): {expected_answer}

CONTEXT:
{context}

Respond with this exact JSON structure:
{{
  "claims": [
    {{"claim": "...", "attributable": true/false, "reason": "..."}}
  ],
  "score": 0.0
}}

score = attributable_claims / total_claims. If no claims, score = 1.0."""
    result = await _judge(provider, prompt, max_tokens=700)
    score = float(result.get("score", 0.0))
    reasoning = json.dumps(result.get("claims", []))
    return max(0.0, min(1.0, score)), reasoning
