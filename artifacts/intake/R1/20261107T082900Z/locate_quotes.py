"""Temporary locator check for legacy quotes. Not a stage task."""
from pathlib import Path

FILES = {
    "log": "research/evidence/legacy-2026-11-07/files/result/s05_dnn.log",
    "pca": "research/evidence/legacy-2026-11-07/files/result/04_embedding_analysis/pca_info.json",
    "meta": "research/evidence/legacy-2026-11-07/files/result/03_embeddings/partial_meta.json",
    "probe": "research/evidence/legacy-2026-11-07/files/result/03_embeddings/partials/routerai__qwen_qwen3-embedding-8b/probe.json",
    "cachehash": "research/evidence/legacy-2026-11-07/files/result/03_embeddings/cache_hash.json",
    "pilot": "research/evidence/legacy-2026-11-07/files/result/pilot_late_acceleration/summary.json",
    "manifest": "research/evidence/legacy-2026-11-07/files/data/source_manifest.json",
    "s03": "src/stages/s03_embeddings.py",
    "cache": "src/cache.py",
    "review": "reviews/04-11-2026.md",
    "protocol": "docs/protocol_revision_2026-11-04.md",
    "listing": "research/stages/R1/inputs/dnn-directory-listing.json",
}
NEEDLES = {
    "log": ["Device: cpu", "noise_regime=clean", "train_acc=1.000", "val_loss=0.0020", "val_loss=0.0038", "width=1024"],
    "pca": ['"final_dims": 4096', "variance_target_not_met_within_cap"],
    "meta": ['"dim": 768'],
    "probe": ['"dim": 4096', "qwen/qwen3-embedding-8b"],
    "cachehash": ["6135da08b14f8ae4"],
    "pilot": ['"embedding_corpus": "not_read"', '"verdict": "inconclusive"', "coarse_peak не означает double descent"],
    "manifest": ["andythetechnerd03/AI-human-text", "Not a byte copy of the Kaggle CSV."],
    "s03": [
        "first_text_hash=hashlib.sha256(all_texts[0].encode()).hexdigest()[:8]",
        'meta.get("hash") == cache_hash',
        '"backend": cfg.EMBEDDING_BACKEND',
    ],
    "cache": ["return _stable_hash(*sorted(kw.items()))"],
    "review": ["В качестве входа используются замороженные эмбеддинги EmbeddingGemma."],
    "protocol": ["Ревью обсуждает EmbeddingGemma. Эмбеддер не переключён."],
    "listing": ['"entries": []'],
}
ROOT = Path(__file__).resolve().parents[4]
for key, relative in FILES.items():
    lines = (ROOT / relative).read_text(encoding="utf-8").splitlines()
    print("==", key)
    for needle in NEEDLES[key]:
        hits = [str(index + 1) for index, line in enumerate(lines) if needle in line]
        print(len(hits), ",".join(hits[:8]), needle)
