import pytest
from urdu_rag.core.evaluation import RetrievalEvaluator

@pytest.fixture
def evaluator():
    return RetrievalEvaluator(k_values=[1, 3, 5])

def test_recall_perfect(evaluator):
    assert evaluator.compute_recall_at_k([1, 2, 3], [1, 2], 3) == 1.0

def test_recall_partial(evaluator):
    assert evaluator.compute_recall_at_k([1, 4, 5], [1, 2], 3) == 0.5

def test_mrr_first_rank(evaluator):
    assert evaluator.compute_mrr([10, 20, 30], [10]) == 1.0

def test_mrr_second_rank(evaluator):
    assert evaluator.compute_mrr([5, 10, 20], [10]) == 0.5

def test_ndcg_perfect_ordering(evaluator):
    # Retrieved: [High, Low] vs Ideal: [High, Low]
    score = evaluator.compute_ndcg_at_k([2.0, 0.0], [2.0, 0.0], 2)
    assert score == 1.0

def test_ndcg_penalized_ordering(evaluator):
    # Retrieved: [Low, High] vs Ideal: [High, Low]
    score = evaluator.compute_ndcg_at_k([0.0, 2.0], [2.0, 0.0], 2)
    assert score < 1.0  # Must be penalized