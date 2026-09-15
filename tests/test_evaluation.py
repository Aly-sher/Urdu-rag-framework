import logging
from urdu_rag.core.evaluation import RetrievalEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Initialize the evaluator (Standard IR k-values)
evaluator = RetrievalEvaluator(k_values=[1, 3, 5])

# 2. Simulate a retrieval scenario
# Let's say we have 2 queries.
# Query 1: "پاکستان کا دارالحکومت" (Capital of Pakistan)
# Query 2: "لاہور کی مشہور غذائیں" (Famous foods of Lahore)

# The IDs of the documents our system actually retrieved (in order of ranking)
predictions = {
    "q1": [101, 102, 105, 108, 110], # Doc 102 is the correct one, but it's ranked 2nd!
    "q2": [201, 202, 203, 204, 205]  # Doc 201 is the correct one, ranked 1st!
}

# The ground truth: Which document IDs are actually relevant?
ground_truth = {
    "q1": [102, 109], # Docs 102 and 109 are relevant
    "q2": [201]       # Only Doc 201 is relevant
}

# Grading the relevance (for nDCG). 2.0 = Highly Relevant, 1.0 = Partially Relevant, 0.0 = Irrelevant
relevances_map = {
    "q1": {102: 2.0, 109: 1.0, 101: 0.0, 105: 0.0, 108: 0.0, 110: 0.0},
    "q2": {201: 2.0, 202: 0.0, 203: 0.0, 204: 0.0, 205: 0.0}
}

# 3. Run the evaluation
logger.info("Running Retrieval Evaluation...")
metrics = evaluator.evaluate(predictions, ground_truth, relevances_map)

print("\n🏆 Retrieval Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"   {metric}: {value:.4f}")