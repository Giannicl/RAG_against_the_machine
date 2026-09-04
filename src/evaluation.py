"""Evaluation: recall of retrieved sources against a ground truth."""

from typing import Dict, List

from .errors import InputError
from .models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


_IOU_THRESHOLD = 0.05
_RECALL_CUTOFFS = (1, 3, 5, 10)
_REPORT_RULE = "=" * 40


def _span_iou_calculator(
    reference_source: MinimalSource, retrieved_source: MinimalSource
) -> float:
    """Compute the IoU of two character spans.

    Args:
        reference_source: Ground-truth source span.
        retrieved_source: Retrieved source span.

    Returns:
        Intersection length over union length of the two spans;
        0.0 when the union is empty.
    """
    intersection = min(
        reference_source.last_character_index,
        retrieved_source.last_character_index,
    ) - max(
        reference_source.first_character_index,
        retrieved_source.first_character_index,
    )
    if intersection <= 0:
        return 0.0
    union = max(
        reference_source.last_character_index,
        retrieved_source.last_character_index,
    ) - min(
        reference_source.first_character_index,
        retrieved_source.first_character_index,
    )
    return intersection / union


def _source_matcher(
    reference_source: MinimalSource,
    retrieved_sources: List[MinimalSource],
) -> bool:
    """Decide whether a ground-truth source was retrieved.

    Args:
        reference_source: Ground-truth source to look for.
        retrieved_sources: The student's retrieved sources.

    Returns:
        True when a retrieved source shares the reference's file path
        and overlaps its span with an IoU above the threshold.
    """
    return any(
        retrieved_source.file_path == reference_source.file_path
        and _span_iou_calculator(reference_source, retrieved_source)
        > _IOU_THRESHOLD
        for retrieved_source in retrieved_sources
    )


def _recall_calculator(
    reference_sources: List[MinimalSource],
    retrieved_sources: List[MinimalSource],
) -> float:
    """Compute the share of ground-truth sources that were retrieved.

    Args:
        reference_sources: Ground-truth sources of one question.
        retrieved_sources: The student's retrieved sources for it.

    Returns:
        Matched reference sources over total reference sources;
        0.0 when the question has no reference sources.
    """
    if not reference_sources:
        return 0.0
    matched = sum(
        _source_matcher(reference_source, retrieved_sources)
        for reference_source in reference_sources
    )
    return matched / len(reference_sources)


def _answered_questions_extractor(
    dataset: RagDataset,
) -> List[AnsweredQuestion]:
    """Keep only the questions that carry ground-truth sources.

    Args:
        dataset: Validated ground-truth dataset.

    Returns:
        The answered questions, in dataset order.
    """
    return [
        rag_question
        for rag_question in dataset.rag_questions
        if isinstance(rag_question, AnsweredQuestion)
    ]


def _report_builder(
    mean_recalls: Dict[int, float], evaluated: int, missing: int
) -> str:
    """Format recall figures the way the grader reports them.

    Args:
        mean_recalls: Mean recall per cutoff.
        evaluated: Number of ground-truth questions evaluated.
        missing: Ground-truth questions absent from student results.

    Returns:
        The printable evaluation report.
    """
    lines = [
        "Evaluation Results",
        _REPORT_RULE,
        f"Questions evaluated: {evaluated}",
    ]
    if missing:
        lines.append(f"Questions missing from student results: {missing}")
    lines.extend(
        f"Recall@{cutoff}: {mean_recall:.3f} ({mean_recall * 100:.1f}%)"
        for cutoff, mean_recall in mean_recalls.items()
    )
    return "\n".join(lines)


def evaluator(
    student_search_results: StudentSearchResults, dataset: RagDataset
) -> str:
    """Score retrieved sources against ground truth at every cutoff.

    Args:
        student_search_results: The student's saved search results.
        dataset: Ground-truth dataset holding answered questions.

    Returns:
        A printable report with the mean recall at each cutoff.

    Raises:
        InputError: The ground truth holds no answered questions.
    """
    answered_questions = _answered_questions_extractor(dataset)
    if not answered_questions:
        raise InputError(
            "no answered questions in the ground-truth dataset"
        )
    retrieved_by_question_id = {
        search_result.question_id: search_result.retrieved_sources
        for search_result in student_search_results.search_results
    }
    missing = sum(
        answered_question.question_id not in retrieved_by_question_id
        for answered_question in answered_questions
    )
    mean_recalls = {
        cutoff: sum(
            _recall_calculator(
                answered_question.sources,
                retrieved_by_question_id.get(
                    answered_question.question_id, []
                )[:cutoff],
            )
            for answered_question in answered_questions
        )
        / len(answered_questions)
        for cutoff in _RECALL_CUTOFFS
    }
    return _report_builder(mean_recalls, len(answered_questions), missing)
