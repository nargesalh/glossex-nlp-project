import json
import numpy as np
from typing import Dict, List


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_seed_list(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def filter_clusters(
    clusters: Dict[str, List[str]],
    embeddings: Dict[str, List[float]],
    econ_seeds: List[str],
    general_seeds: List[str],
) -> Dict[str, List[str]]:
    filtered = {}

    for cid, tokens in clusters.items():
        econ_scores = []
        gen_scores = []

        for token in tokens:
            v = np.array(embeddings[token])

            for s in econ_seeds:
                if s in embeddings:
                    econ_scores.append(
                        cosine_similarity(v, np.array(embeddings[s]))
                    )

            for s in general_seeds:
                if s in embeddings:
                    gen_scores.append(
                        cosine_similarity(v, np.array(embeddings[s]))
                    )

        if econ_scores and gen_scores and np.mean(econ_scores) > np.mean(gen_scores):
            filtered[cid] = tokens

    return filtered


def main():
    with open("data/processed/clusters.json", "r") as f:
        clusters = json.load(f)

    with open("data/processed/lemma_embeddings.json", "r") as f:
        embeddings = json.load(f)

    econ_seeds = load_seed_list("data/seeds/economics.txt")
    general_seeds = load_seed_list("data/seeds/general.txt")

    filtered = filter_clusters(
        clusters, embeddings, econ_seeds, general_seeds
    )

    with open("data/processed/final_terms.json", "w") as f:
        json.dump(filtered, f, indent=2)

    print(
        f"Filtering completed. "
        f"{len(filtered)} clusters selected as economics-related."
    )


if __name__ == "__main__":
    main()
