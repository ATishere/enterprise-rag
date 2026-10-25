# Enterprise RAG Knowledge Base Platform

A production-oriented retrieval-augmented generation demo combining hybrid search, graph retrieval, reranking, access control, citations, and RAGAS evaluation.

## Features

- Hybrid PostgreSQL full-text and pgvector retrieval fused with reciprocal rank fusion.
- Optional Neo4j multi-hop graph retrieval with tenant isolation.
- HyDE query preprocessing and acronym expansion.
- Cross-encoder reranking.
- Tenant and user ACL checks in retrieval SQL and a second check before generation.
- Citation validation against answer spans and retrieved chunk IDs.
- RAGAS evaluation with five metrics and a faithfulness CI hard gate.
- Graceful degradation when optional graph or reranking services fail.

## Quick start

Docker Compose starts the API and local PostgreSQL/pgvector and Neo4j services:

```bash
docker compose up -d --build
```

Check API readiness:

```bash
curl http://localhost:8000/health
```

The local Neo4j Browser is available at `http://localhost:7475` and Bolt at `localhost:7688` by default. Override `NEO4J_HTTP_PORT` or `NEO4J_BOLT_PORT` if these host ports are occupied.

The first query downloads the embedding and reranker models. Answer generation also requires `ANTHROPIC_API_KEY`; set it in the environment or a local `.env` file before starting Compose. Local database credentials are demo defaults only; override `POSTGRES_PASSWORD` and `NEO4J_PASSWORD` outside local development.

## Evaluate

Add `OPENAI_API_KEY` for the RAGAS judge and `ANTHROPIC_API_KEY` for answer generation to `.env`, then seed the golden examples and run the evaluation inside the API container:

```bash
docker compose exec api python scripts/seed_eval_data.py
docker compose exec api python scripts/run_eval.py --hard-gate faithfulness
```

Faithfulness below `0.90` fails the command. Answer relevancy, context precision, and context recall are reported as soft warnings. Evaluation makes paid model calls; the pull-request evaluation job runs only when both GitHub secrets are configured. Forked pull requests without secrets skip this paid job.

## Architecture

Client → FastAPI → preprocessing → PostgreSQL hybrid search → optional Neo4j graph search → cross-encoder reranking → ACL re-check → answer generation.

## CI

Every push and pull request to `main` runs Ruff, MyPy, and unit tests. Current measured coverage is about 58%, with a 55% CI floor; raise the coverage threshold as test coverage grows. Pull requests can additionally run the RAGAS evaluation when the required secrets are available.

## Known production hardening

- `tenant_id` and `user_id` are currently supplied in the request body. A deployed service must derive them from an authenticated identity (for example, a validated JWT), not trust client-provided values.
- Streaming errors after response headers are sent are returned as an SSE `event: error`, not an HTTP error status.
- Add rate limiting, request limits, and operational monitoring before public deployment.
- Replace local demo database credentials and configure secret management for deployment.
