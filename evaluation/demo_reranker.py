"""
Fully offline demo of the cross-encoder reranker (no Pinecone / no API key).

Shows how the cross-encoder re-scores a fixed set of candidate passages for a
medical query, surfacing the truly relevant ones above lexically-similar noise.
This is the script behind the sample output in the README.

Usage:  python -m evaluation.demo_reranker
"""
from src.reranker import rerank_passages

QUERY = "What are the common symptoms of asthma?"

# Mixed-relevance candidates a bi-encoder might all return for this query.
CANDIDATES = [
    "Asthma symptoms include wheezing, shortness of breath, chest tightness, "
    "and coughing that is often worse at night or early in the morning.",
    "Asthma is a chronic disease of the airways and affects millions of people "
    "of all ages worldwide.",
    "Treatment of asthma involves inhaled corticosteroids and quick-relief "
    "bronchodilators such as albuterol.",
    "Diabetes mellitus is characterized by high blood glucose due to defects "
    "in insulin secretion or action.",
    "During an asthma attack the airways narrow and swell and produce extra "
    "mucus, making it hard to breathe and triggering wheezing and coughing.",
    "Regular exercise and a balanced diet are important for overall "
    "cardiovascular health.",
]


def main():
    print("QUERY:", QUERY)
    print("\nCandidates (bi-encoder would return all of these):")
    for i, c in enumerate(CANDIDATES, 1):
        print(f"  {i}. {c[:80].strip()}...")

    results = rerank_passages(QUERY, CANDIDATES, top_n=3)
    print("\nCross-encoder reranked TOP 3:")
    for r in results:
        print(f"  rank {r['rank']}  score={r['score']:.3f}  {r['text'][:80].strip()}...")


if __name__ == "__main__":
    main()
