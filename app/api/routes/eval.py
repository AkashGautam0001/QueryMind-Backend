import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.clients import get_arq_pool
from app.core.logging import get_logger
from app.db.session import get_db
from app.models.eval_question import EvalQuestion
from app.models.evaluation import EvalRun, EvalRunStatus
from app.schemas.evaluation import (
    EvalQuestionCreate,
    EvalQuestionResponse,
    EvalRunDetailResponse,
    EvalRunResponse,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.post("/questions", status_code=201, response_model=EvalQuestionResponse)
async def create_question(
    payload: EvalQuestionCreate, session: AsyncSession = Depends(get_db)
) -> EvalQuestion:
    question = EvalQuestion(
        question=payload.question,
        expected_answer=payload.expected_answer,
        document_id=payload.document_id,
    )
    session.add(question)
    await session.commit()
    await session.refresh(question)
    return question


@router.get("/questions", response_model=list[EvalQuestionResponse])
async def list_questions(session: AsyncSession = Depends(get_db)) -> list[EvalQuestion]:
    result = await session.execute(select(EvalQuestion).order_by(EvalQuestion.created_at))
    return list(result.scalars().all())


@router.delete("/questions/{question_id}", status_code=204)
async def delete_question(
    question_id: uuid.UUID, session: AsyncSession = Depends(get_db)
) -> None:
    question = await session.get(EvalQuestion, question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")
    await session.delete(question)
    await session.commit()


@router.post("/runs", status_code=202, response_model=EvalRunResponse)
async def trigger_run(session: AsyncSession = Depends(get_db)) -> EvalRun:
    # Require at least one question before accepting a run — an empty run
    # would succeed trivially and its zero-question aggregate scores would
    # pollute the trend line without indicating anything real.
    count_result = await session.execute(select(EvalQuestion))
    if not list(count_result.scalars().all()):
        raise HTTPException(
            status_code=422,
            detail="Add at least one eval question before triggering a run.",
        )

    run = EvalRun()
    session.add(run)
    await session.flush()
    await session.commit()

    pool = await get_arq_pool()
    await pool.enqueue_job("run_eval", str(run.id))

    logger.info("eval_run_triggered", run_id=str(run.id))
    await session.refresh(run)
    return run


@router.get("/runs", response_model=list[EvalRunResponse])
async def list_runs(session: AsyncSession = Depends(get_db)) -> list[EvalRun]:
    result = await session.execute(
        select(EvalRun).order_by(EvalRun.created_at.desc()).limit(50)
    )
    return list(result.scalars().all())


@router.get("/runs/{run_id}", response_model=EvalRunDetailResponse)
async def get_run(run_id: uuid.UUID, session: AsyncSession = Depends(get_db)) -> EvalRun:
    result = await session.execute(
        select(EvalRun)
        .where(EvalRun.id == run_id)
        .options(selectinload(EvalRun.results))
    )
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return run
