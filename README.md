# VoiceReach AI

A retrieval-augmented voice assistant backend that helps underserved Nigerian communities get timely, grounded health and community information over voice calls. VoiceReach AI accepts transcribed caller speech, classifies the situation, retrieves grounded knowledge, generates a safety-checked reply, and returns a structured JSON response for a text-to-speech layer to speak back to the caller.

## Why this exists
Many callers in low-bandwidth / voice-first contexts need reliable, evidence-based answers (health guidance, symptom triage, referrals). VoiceReach AI combines LLM-based assessment, a small deterministic dialogue manager, and a persistent knowledge retrieval layer (RAG) to produce audit-able, grounded spoken replies while protecting against unsafe or hallucinated responses.

## Quick highlights
- Strong separation of concerns: assessment LLM → dialogue decision → RAG retrieval → response LLM → safety validator.
- Typed domain models (Pydantic v2) used to validate LLM outputs and API responses.
- Ingest CLI to populate a ChromaDB knowledge collection from local files.
- Designed to degrade gracefully (fallback responses) when networks/LLM calls fail.

---

## Architecture (high level)

User Speech (STT upstream)
  ↓
Orchestrator.handle_message()
  - Situation Assessment (LLM call 1 → structured JSON)
  - Dialogue decision (emergency / clarify / retrieve)
  - Knowledge Retrieval (ChromaDB RAG)
  - Response Generation (LLM call 2)
  - Safety Validation
  ↓
AIResponse JSON → TTS / telephony layer

Every request flows through `app.services.orchestrator.Orchestrator.handle_message()` which sequences the pipeline and persists conversation state.

---

## Stack
- Language: Python 3.12+
- Framework/runtime: FastAPI (ASGI server via uvicorn)
- Notable libraries:
  - Pydantic v2 (typed models/settings)
  - Chroma / sentence-transformers (RAG embeddings & retrieval)
  - OpenAI-compatible LLM gateway (configurable; current defaults reference Gemini/OpenAI-compatible endpoint)

---

## Repository layout

```
app/
  main.py                     FastAPI application entry point
  core/
    config.py                  Typed settings (pydantic-settings, .env support)
    models.py                  Domain models and JSON contracts (Pydantic)
  prompts/
    prompts.py                 Prompt builders for assessment & response
  services/
    llm_service.py             LLM gateway (assess_situation, generate_response)
    orchestrator.py            Orchestrator (pipeline sequencing)
    knowledge.py               ChromaDB-backed retrieval (RAG)
    safety.py                  Safety/grounding validator
    conversation_store.py      In-memory ConversationState persistence (swapable)
  api/
    dependencies.py            FastAPI DI providers
    routes/chat.py             POST /chat route
  utils/
    logging.py                 Structured logging config
scripts/
  ingest.py                    CLI to ingest data/* -> Chroma
data/
  knowledge/                   Example knowledge docs (JSON/MD/TXT/CSV)
  chroma/                      Chroma persistence (generated)
requirements.txt
.env.example
```

How it fits together:
- The HTTP API (`app.main`) exposes POST /chat which calls the Orchestrator.
- The Orchestrator uses `LLMService` to classify the incoming text (strict JSON `SituationAssessment`), decides the next stage, optionally runs the RAG retriever, asks the response LLM to produce a reply, and finally runs the `SafetyValidator` before returning an `AIResponse`.
- `ConversationState` (in-memory by default) is the single source of truth for session context; it should be swapped to a durable store for production.

---

## Setup (development)

Prerequisites: Python 3.12+, Git

```bash
git clone https://github.com/darish500/VoiceReachAI.git
cd VoiceReachAI
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env to set required env vars (OPENAI_API_KEY and others)
```

Settings are typed in `app/core/config.py` and load from `.env` by default.

---

## Environment variables

Use the `.env` (copy from `.env.example`) to configure runtime behavior. Key variables:

- OPENAI_API_KEY — OpenAI / provider API key (required)
- LLM_BASE_URL — Base URL for OpenAI-compatible endpoint (default set in config)
- ASSESSMENT_MODEL — Model used for Situation Assessment (default: `gemini-2.5-flash`)
- RESPONSE_MODEL — Model used for Response Generation (default: `gemini-2.5-flash`)
- ASSESSMENT_TEMPERATURE — Temperature for assessment model (default: `0.0`)
- RESPONSE_TEMPERATURE — Temperature for response model (default: `0.4`)
- LLM_TIMEOUT_SECONDS — Timeout for external LLM calls (default: `20.0`)
- RETRIEVAL_TOP_K — Number of chunks to retrieve from RAG (default: `3`)
- CHROMA_PERSIST_DIRECTORY — On-disk directory used by Chroma (default: `data/chroma`)
- CHROMA_COLLECTION_NAME — Chroma collection name (default: `voice_reach_knowledge`)
- EMBEDDING_MODEL — Sentence-transformers model for embeddings (default: `all-MiniLM-L6-v2`)
- ENVIRONMENT — `development` / `production`
- LOG_LEVEL — Logging verbosity

Refer to `app/core/config.py` for the canonical defaults and typed values.

---

## Ingest knowledge

Add files (.json, .md, .txt, .csv) to `data/knowledge/` and run:

```bash
python -m scripts.ingest --source data/knowledge
```

This will convert documents into `KnowledgeChunk` objects and insert them into the Chroma collection identified by `CHROMA_PERSIST_DIRECTORY` / `CHROMA_COLLECTION_NAME`.

---

## Run the API

Development server (reload enabled):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API root: http://localhost:8000
- Swagger UI (interactive docs): http://localhost:8000/docs

---

## Example API: POST /chat

Request:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
        "session_id": "caller-123",
        "message": "My child has had a fever for two days, what should I do?"
      }'
```

Example AIResponse (structured JSON):

```json
{
  "message": "A fever lasting two days in a child may be a sign of malaria or another infection. Please take your child to the nearest clinic today for testing, and seek urgent care if they have difficulty breathing, persistent vomiting, or are unusually drowsy.",
  "intent": "symptom_check",
  "stage": "providing_answer",
  "risk_level": "medium",
  "continue_conversation": true,
  "confidence": 0.87,
  "sources": ["malaria_symptoms_001"]
}
```

Notes:
- The first LLM call (Situation Assessment) returns a strict JSON contract (`SituationAssessment`) which the orchestrator validates before acting.
- The `sources` list contains KnowledgeChunk IDs used to ground the reply (only IDs, not full content).

---

## Testing & local development notes
- Unit-test the Orchestrator helpers: `_decide_next_step`, `_merge_assessment`, and `_retrieve_knowledge` can be exercised in isolation.
- Replace the in-memory `ConversationStore` with a Redis/Postgres-backed store for multi-instance production usage.
- Add automated regression tests using recorded transcripts to detect changes in the LLM prompt/responses.

---

## Production considerations & future work
- Replace in-memory conversation persistence with a durable store (Redis/Postgres).
- Add authentication, rate limiting, and telemetry at the API layer.
- Promote the inlined dialogue branching to a configurable `DialogueManager`.
- Add multi-turn evaluation harness and automated regression tests.
- Consider streaming or chunked responses to reduce perceived latency for long replies.

---

## Contributing
Contributions welcome — open an issue describing the feature or bug, then submit a PR. Please:
- Keep the LLM prompts and API contracts stable where possible.
- Add unit tests for new logic in `app/services`.
- Update `.env.example` and README if you introduce new top-level settings.

---

## License
Add your chosen license here (e.g. MIT). If you want me to add a license file, I can prepare one.

---

## Maintainers / contact
Repository: https://github.com/darish500/VoiceReachAI

If you'd like, I can:
- Convert this README into a pull request and update the repository file,
- Or trim/expand any section (API docs, architecture diagram, or contributor guidance).
