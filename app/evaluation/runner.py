import asyncio
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import get_llm_provider
from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal
from app.evaluation.metrics import (
    score_answer_relevancy,
    score_context_precision,
    score_context_recall,
    score_faithfulness,
)
from app.generation.pipeline import NO_RELEVANT_CONTEXT_MESSAGE
from app.models.eval_question import EvalQuestion
from app.models.evaluation import EvalResult, EvalRun, EvalRunStatus
from app.retrieval.context import assemble_context, retrieve_and_rerank

logger = get_logger(__name__)


async def _run_question(
    session: AsyncSession,
    run_id: uuid.UUID,
    question: EvalQuestion,
) -> EvalResult:
    """Runs the full retrieve → generate → score pipeline for one question."""
    provider = get_llm_provider()

    reranked = await retrieve_and_rerank(
        query=question.question,
        session=session,
        document_id=question.document_id,
    )
    assembled = assemble_context(reranked)
    context_chunks = [c.content for c in reranked]
    context_text = assembled.context_text

    if not reranked:
        generated_answer = NO_RELEVANT_CONTEXT_MESSAGE
    else:
        from app.generation.prompt import build_system_prompt

        messages = [{"role": "user", "content": question.question}]
        system_prompt = build_system_prompt(context_text)
        parts = []
        async for delta in provider.stream(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=512,
            temperature=0.0,
        ):
            parts.append(delta)
        generated_answer = "".join(parts)

    # All four metrics run concurrently — they're independent judge calls.
    (faith_score, faith_reason), (rel_score, rel_reason), (prec_score, prec_reason), (rec_score, rec_reason) = await asyncio.gather(
        score_faithfulness(provider, question.question, generated_answer, context_text),
        score_answer_relevancy(provider, question.question, generated_answer),
        score_context_precision(provider, question.question, generated_answer, context_chunks),
        score_context_recall(provider, question.question, question.expected_answer, context_text),
    )

    return EvalResult(
        run_id=run_id,
        question_id=question.id,
        generated_answer=generated_answer,
        retrieved_context=[{"content": c, "filename": r.filename, "page": r.page_number}
                           for c, r in zip(context_chunks, reranked, strict=False)],
        faithfulness=faith_score,
        answer_relevancy=rel_score,
        context_precision=prec_score,
        context_recall=rec_score,
        judge_reasoning={
            "faithfulness": faith_reason,
            "answer_relevancy": rel_reason,
            "context_precision": prec_reason,
            "context_recall": rec_reason,
        },
    )


async def run_evaluation(run_id: uuid.UUID) -> None:
    """Executes a full evaluation run. Called by the arq worker."""
    async with AsyncSessionLocal() as session:
        run = await session.get(EvalRun, run_id)
        if run is None:
            logger.error("eval_run_not_found", run_id=str(run_id))
            return
        run.status = EvalRunStatus.RUNNING
        run.started_at = datetime.now(UTC)
        await session.commit()

    try:
        async with AsyncSessionLocal() as session:
            result_rows = await session.execute(select(EvalQuestion))
            questions = list(result_rows.scalars().all())

            if not questions:
                raise ValueError("No eval questions found. Add some via POST /eval/questions first.")

            results = []
            for question in questions:
                try:
                    result = await _run_question(session, run_id, question)
                    session.add(result)
                    results.append(result)
                    await session.flush()
                except Exception as exc:
                    logger.error(
                        "eval_question_failed",
                        question_id=str(question.id),
                        error=str(exc),
                    )
                    # Continue with remaining questions rather than aborting the whole run.

            if results:
                run = await session.get(EvalRun, run_id)
                run.mean_faithfulness = sum(r.faithfulness for r in results) / len(results)
                run.mean_answer_relevancy = sum(r.answer_relevancy for r in results) / len(results)
                run.mean_context_precision = sum(r.context_precision for r in results) / len(results)
                run.mean_context_recall = sum(r.context_recall for r in results) / len(results)
                run.question_count = len(results)
                run.status = EvalRunStatus.COMPLETED
                run.completed_at = datetime.now(UTC)
                await session.commit()

                logger.info(
                    "eval_run_completed",
                    run_id=str(run_id),
                    questions=len(results),
                    faithfulness=round(run.mean_faithfulness, 3),
                    answer_relevancy=round(run.mean_answer_relevancy, 3),
                    context_precision=round(run.mean_context_precision, 3),
                    context_recall=round(run.mean_context_recall, 3),
                )

    except Exception as exc:
        logger.error("eval_run_failed", run_id=str(run_id), error=str(exc))
        async with AsyncSessionLocal() as session:
            run = await session.get(EvalRun, run_id)
            if run:
                run.status = EvalRunStatus.FAILED
                run.error_message = str(exc)
                run.completed_at = datetime.now(UTC)
                await session.commit()
