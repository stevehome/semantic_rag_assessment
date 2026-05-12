"""Sample technical corpus + labelled relevance set.

Documents are intentionally long enough (~70-100 tokens each) that the token
chunker produces 2-3 chunks per doc — exercising the parent_doc_id
deduplication path in evaluation. Vocabulary deliberately varies so query
expansion can do useful work (e.g. "peak load" never appears verbatim).
"""

DOCUMENTS = [
    {
        "id": "doc-01",
        "title": "Autoscaling and traffic spikes",
        "text": (
            "The platform reacts to traffic spikes by horizontally scaling stateless "
            "services behind an L7 load balancer. A predictive autoscaler watches "
            "request-per-second and CPU utilisation and pre-warms instances before "
            "thresholds are breached, which keeps tail latency stable during bursts. "
            "During regional failover the same control loop drains a zone and "
            "redistributes capacity within seconds, so the user-visible impact of a "
            "single-zone outage is a brief uptick in p95 rather than a hard error rate."
        ),
    },
    {
        "id": "doc-02",
        "title": "Cold start mitigation",
        "text": (
            "To avoid cold starts on freshly provisioned pods, the orchestrator runs a "
            "lightweight warmup probe that primes JIT caches and database connection "
            "pools. This shaves several hundred milliseconds off the first request. "
            "For functions invoked rarely, a periodic synthetic heartbeat keeps at "
            "least one warm replica in rotation; the cost is negligible compared to "
            "the latency tax users would otherwise see on the cold path."
        ),
    },
    {
        "id": "doc-03",
        "title": "Caching strategy",
        "text": (
            "A two-tier cache absorbs read amplification: an in-process LRU sits in "
            "front of a shared Redis cluster. Write-through invalidation keeps the "
            "tiers coherent, and consistent hashing prevents hot-key stampedes by "
            "spreading load across nodes. Short TTLs on the L1 tier protect against "
            "stale reads after writes; long TTLs on the L2 tier amortise cold misses "
            "across many requests."
        ),
    },
    {
        "id": "doc-04",
        "title": "Vector index choice",
        "text": (
            "For semantic retrieval we evaluated HNSW, IVF-PQ and brute-force flat "
            "indexes. HNSW gave the best recall/latency trade-off at our scale "
            "(under 10 million vectors), while IVF-PQ is reserved for the cold "
            "archive where memory pressure matters more than tail latency. The "
            "evaluation was driven by recall@10 against a labelled query set, "
            "controlling for index build time and memory footprint."
        ),
    },
    {
        "id": "doc-05",
        "title": "Embedding model selection",
        "text": (
            "We use a dense sentence encoder fine-tuned on technical documentation. "
            "The encoder maps queries and passages into a shared 384-dimensional "
            "space; cosine similarity is the ranking signal because the model was "
            "trained with a cosine contrastive objective. A reranker runs over the "
            "top-50 ANN candidates to recover precision lost to ANN approximation."
        ),
    },
    {
        "id": "doc-06",
        "title": "Observability and SLOs",
        "text": (
            "Each retrieval call emits a structured trace with span attributes for "
            "embedding latency, ANN search latency and rerank latency. SLOs are set "
            "on p95 end-to-end latency and on recall@10 measured against a labelled "
            "evaluation set. Alerts fire on both the latency dashboard and on "
            "recall regression — a quiet quality drop is more dangerous than a loud "
            "latency spike because users do not see it directly."
        ),
    },
    {
        "id": "doc-07",
        "title": "Failure isolation",
        "text": (
            "Circuit breakers around the embedding service prevent a slow encoder "
            "from cascading into upstream timeouts. On breaker-open we fall back to "
            "a BM25 lexical retriever so the product stays useful, just less "
            "semantic. The fallback path is exercised in staging weekly so it does "
            "not bit-rot — a fallback that has not been tested is not a fallback."
        ),
    },
    {
        "id": "doc-08",
        "title": "Data ingestion pipeline",
        "text": (
            "Raw documents are chunked into roughly 200-token windows with a "
            "40-token overlap. Each chunk inherits metadata from its parent "
            "document (source, timestamp, ACL) so retrieval results can be filtered "
            "post-hoc. Chunks are embedded in batches and upserted into the vector "
            "index; deletes are tombstoned in the metadata store and compacted "
            "during the nightly rebuild."
        ),
    },
]


# Labelled relevance — used by src/eval.py. Mapping is at parent-document
# granularity; predictions are deduplicated to parent_doc_id before scoring.
LABELLED_QUERIES = [
    # Paraphrased queries (encoder should win, BM25 should struggle).
    {"query": "How does the system handle peak load?",
     "relevant_doc_ids": {"doc-01"}},
    {"query": "What happens when the embedding service is slow or fails?",
     "relevant_doc_ids": {"doc-07"}},
    {"query": "How do you monitor retrieval quality in production?",
     "relevant_doc_ids": {"doc-06"}},
    {"query": "Which caching layers are used?",
     "relevant_doc_ids": {"doc-03"}},
    {"query": "How is cold latency mitigated for new pods?",
     "relevant_doc_ids": {"doc-02"}},
    {"query": "How is the corpus broken up before indexing?",
     "relevant_doc_ids": {"doc-08"}},
    {"query": "Why is the platform resilient to single-zone failures?",
     "relevant_doc_ids": {"doc-01"}},
    {"query": "What is the fallback when the encoder times out?",
     "relevant_doc_ids": {"doc-07"}},
    # Multi-relevant queries.
    {"query": "How are documents prepared for semantic search?",
     "relevant_doc_ids": {"doc-08", "doc-04"}},
    {"query": "What signals are used to rank passages?",
     "relevant_doc_ids": {"doc-05", "doc-04"}},
    # Lexical / proper-noun queries (BM25 should help — Strategy C territory).
    {"query": "HNSW vs IVF-PQ tradeoffs",
     "relevant_doc_ids": {"doc-04"}},
    {"query": "Redis cache invalidation",
     "relevant_doc_ids": {"doc-03"}},
    {"query": "BM25 fallback retriever",
     "relevant_doc_ids": {"doc-07"}},
    {"query": "p95 latency SLO",
     "relevant_doc_ids": {"doc-06"}},
    {"query": "JIT cache warmup probe",
     "relevant_doc_ids": {"doc-02"}},
    {"query": "Which ANN index did you pick and why?",
     "relevant_doc_ids": {"doc-04"}},
]
