import pytest
from urdu_rag.core.hard_negatives import BM25HardNegativeMiner

@pytest.fixture
def miner():
    corpus = [
        "اسلام آباد پاکستان کا دارالحکومت ہے۔",
        "کراچی پاکستان کا سب سے بڑا شہر ہے۔",
        "لاہور پنجاب کا دارالحکومت ہے۔",
        "پاکستان کا قومی کھیل ہاکی ہے۔"
    ]
    return BM25HardNegativeMiner(corpus)

def test_mining_returns_correct_count(miner):
    queries = ["پاکستان کا دارالحکومت"]
    positives = ["اسلام آباد پاکستان کا دارالحکومت ہے۔"]
    triplets = miner.mine_triplets(queries, positives, k_negatives=2)
    assert len(triplets) == 2

def test_positive_not_in_negatives(miner):
    queries = ["پاکستان کا دارالحکومت"]
    positive = "اسلام آباد پاکستان کا دارالحکومت ہے۔"
    positives = [positive]
    triplets = miner.mine_triplets(queries, positives, k_negatives=3)
    
    for q, p, n in triplets:
        assert p == positive
        assert n != positive, "Hard negative cannot be the positive document!"