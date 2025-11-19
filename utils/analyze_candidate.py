import asyncio
import json
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from utils.llm_call import get_response_from_llm, parse_json_response
from utils.prompts import next_question_generation, feedback_generation

executor = ThreadPoolExecutor(max_workers=4)


class InterviewAnalysisError(Exception):
    """Custom exception for interview analysis errors"""
    pass


@lru_cache(maxsize=128)
def _cache_key(prompt: str) -> str:
    return hash(prompt)


def _safe_json_parse(data: Any) -> Dict[str, Any]:
    """
    Safely parse LLM output and guarantee a dictionary.
    """
    if data is None:
        raise InterviewAnalysisError("LLM returned None instead of valid JSON")

    if isinstance(data, dict):
        return data

    if isinstance(data, str):
        data = data.strip()

        # Try JSON parse directly
        try:
            return json.loads(data)
        except Exception:
            pass

        # Try to extract JSON from text
        try:
            start = data.index("{")
            end = data.rindex("}") + 1
            json_text = data[start:end]
            return json.loads(json_text)
        except Exception:
            raise InterviewAnalysisError(f"Invalid JSON from LLM: {data}")

    raise InterviewAnalysisError(f"LLM returned unsupported format: {type(data)}")


async def _make_llm_call_async(prompt: str) -> Dict[str, Any]:
    """
    Make LLM call asynchronously and ensure JSON is returned.
    """
    try:
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(executor, get_response_from_llm, prompt)

        response_json = parse_json_response(response_text)

        # Fallback if parse_json_response returns None
        safe_json = _safe_json_parse(response_json)
        return safe_json

    except Exception as e:
        raise InterviewAnalysisError(f"Failed to get LLM response: {str(e)}")


# ---------------------------------------------------------------------------
# NEXT QUESTION
# ---------------------------------------------------------------------------

async def get_next_question(
    previous_question: str,
    candidate_response: str,
    resume_highlights: str,
    job_description: str
) -> str:

    try:
        final_prompt = next_question_generation.format(
            previous_question=previous_question,
            candidate_response=candidate_response,
            resume_highlights=resume_highlights,
            job_description=job_description,
        )

        response = await _make_llm_call_async(final_prompt)

        if not isinstance(response, dict):
            raise InterviewAnalysisError(f"LLM returned non-dict: {response}")

        if "next_question" not in response:
            raise InterviewAnalysisError(
                f"Missing 'next_question' in LLM output: {response}"
            )

        return response["next_question"]

    except Exception as e:
        raise InterviewAnalysisError(f"Question generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# FEEDBACK
# ---------------------------------------------------------------------------

async def get_feedback_of_candidate_response(
    question: str,
    candidate_response: str,
    job_description: str,
    resume_highlights: str
) -> Dict[str, Any]:

    try:
        final_prompt = feedback_generation.format(
            question=question,
            candidate_response=candidate_response,
            job_description=job_description,
            resume_highlights=resume_highlights,
        )

        response = await _make_llm_call_async(final_prompt)

        if not isinstance(response, dict):
            raise InterviewAnalysisError(f"Invalid feedback response: {response}")

        required_fields = ["feedback", "score"]
        missing = [f for f in required_fields if f not in response]
        if missing:
            raise InterviewAnalysisError(f"Missing fields: {missing}")

        # Validate score
        try:
            score = float(response["score"])
            if not (0 <= score <= 10):
                raise InterviewAnalysisError(f"Score out of range: {score}")
        except Exception:
            raise InterviewAnalysisError(f"Score must be a number: {response['score']}")

        return {
            "feedback": response["feedback"],
            "score": response["score"]
        }

    except Exception as e:
        raise InterviewAnalysisError(f"Feedback generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# COMBINED — NEXT QUESTION + FEEDBACK
# ---------------------------------------------------------------------------

async def analyze_candidate_response_and_generate_new_question(
    question: str,
    candidate_response: str,
    job_description: str,
    resume_highlights: str,
    timeout: float = 30.0
) -> Tuple[str, Dict[str, Any]]:

    try:
        # run both tasks in parallel
        feedback_task = get_feedback_of_candidate_response(
            question, candidate_response, job_description, resume_highlights
        )

        next_question_task = get_next_question(
            question, candidate_response, resume_highlights, job_description
        )

        feedback, next_question = await asyncio.wait_for(
            asyncio.gather(feedback_task, next_question_task),
            timeout=timeout
        )

        return next_question, feedback

    except asyncio.TimeoutError:
        raise InterviewAnalysisError("LLM operations timed out")

    except Exception as e:
        raise InterviewAnalysisError(f"Response analysis failed: {str(e)}")
