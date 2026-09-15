import logging
from urdu_rag.core.hard_negatives import BM25HardNegativeMiner

logging.basicConfig(level=logging.INFO)

# 1. Define a small Urdu corpus
corpus = [
    "اسلام آباد پاکستان کا دارالحکومت ہے۔",
    "کراچی پاکستان کا سب سے بڑا شہر اور مالیاتی مرکز ہے۔",
    "لاہور پنجاب کا دارالحکومت اور ثقافتی مرکز ہے۔",
    "پاکستان کا قومی کھیل ہاکی ہے۔",
    "اسلام آباد میں فیصل مسجد واقع ہے جو ایک مشہور سیاحتی مقام ہے۔",
    "کراچی میں سمندر کے کنارے بہت خوبصورت ہیں۔"
]

# 2. Define queries and their actual positive (correct) documents
queries = [
    "پاکستان کا دارالحکومت کون سا شہر ہے؟",
    "لاہور کس صوبے کا دارالحکومت ہے؟"
]

positives = [
    "اسلام آباد پاکستان کا دارالحکومت ہے۔", # Exact match in corpus
    "لاہور پنجاب کا دارالحکومت اور ثقافتی مرکز ہے۔" # Exact match in corpus
]

# 3. Initialize the miner
miner = BM25HardNegativeMiner(corpus)

# 4. Mine triplets
triplets = miner.mine_triplets(queries, positives, k_negatives=2)

print("\n🏆 Mined Hard-Negative Triplets:")
for q, p, n in triplets:
    print(f"Query: {q}")
    print(f"  [+] Positive: {p}")
    print(f"  [-] Hard Neg: {n}\n")