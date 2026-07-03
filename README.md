# VoiceReach AI

A Retrieval-Augmented Voice Assistant backend for underserved Nigerian communities.

VoiceReach AI receives transcribed caller speech, classifies the caller's
situation, retrieves grounded knowledge relevant to the request, generates
a safety-checked spoken reply, and returns it as structured JSON for a
text-to-speech layer to speak back to the caller.

## Architecture

```
User Speech
   |
   v
Speech-to-Text (outside this project)
   |
   v
Orchestrator  ------------------------------------------------+
   |                                                           |
   v                                                           |
Situation Assessment (LLM Call 1)                              |
   |                                                           |
   v                                                           |
Dialogue Decision (emergency / clarify / retrieve)              |
   |                                                           |
   v                                                           |
Knowledge Retrieval (RAG via ChromaDB)                          |
   |                                                           |
   v                                                           |
Response Generation (LLM Call 2)                                |
   |                                                           |
   v                                                           |
Safety Validation  <---------------------------------------------+
   |
   v
JSON Response (AIResponse)
   |
   v
Text-to-Speech (outside this project)
```

Every request flows through exactly one entry point,
`Orchestrator.handle_message()`, which sequences the collaborators above.
Each collaborator (LLMService, KnowledgeRetriever, SafetyValidator,
ConversationStore) is independently swappable behind a narrow interface.

## Folder structure

```
app/
  core/
    config.py          Typed settings (pydantic-settings, .env support)
    models.py           Domain models: Message, ConversationState, AIResponse, etc.
  prompts/
    prompts.py           Prompt construction for both LLM calls
  services/
    llm_service.py        OpenAI gateway (assess_situation, generate_response)
    knowledge.py           ChromaDB-backed RAG retriever
    safety.py               Final safety/grounding validation on every reply
    conversation_store.py    In-memory ConversationState persistence
    orchestrator.py         Coordinates the full pipeline
  api/
    dependencies.py       FastAPI DI providers (singletons)
    routes/chat.py         POST /chat
  utils/
    logging.py               Structured logging configuration
  main.py                    FastAPI application entry point
scripts/
  ingest.py                   Knowledge ingestion CLI (JSON/TXT/MD/CSV -> Chroma)
data/
  knowledge/                   Sample knowledge source files
  chroma/                       Chroma persistence directory (generated)
requirements.txt
.env.example
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | *(required)* |
| `ASSESSMENT_MODEL` | Model for Situation Assessment (Call 1) | `gpt-4o-mini` |
| `RESPONSE_MODEL` | Model for Response Generation (Call 2) | `gpt-4o-mini` |
| `ASSESSMENT_TEMPERATURE` | Sampling temperature, Call 1 | `0.0` |
| `RESPONSE_TEMPERATURE` | Sampling temperature, Call 2 | `0.4` |
| `LLM_TIMEOUT_SECONDS` | Per-request OpenAI timeout | `20.0` |
| `RETRIEVAL_TOP_K` | Chunks retrieved per query | `3` |
| `CHROMA_PERSIST_DIRECTORY` | Chroma on-disk storage path | `data/chroma` |
| `CHROMA_COLLECTION_NAME` | Chroma collection name | `voice_reach_knowledge` |
| `EMBEDDING_MODEL` | sentence-transformers model | `all-MiniLM-L6-v2` |
| `ENVIRONMENT` | `development` / `production` | `development` |
| `LOG_LEVEL` | Python logging level | `INFO` |

## Ingesting knowledge documents

Place `.json`, `.txt`, `.md`, or `.csv` files under `data/knowledge/`
(a sample `malaria.json` is included), then run:

```bash
python -m scripts.ingest --source data/knowledge
```

This loads each file, converts it into `KnowledgeChunk` objects, and
inserts them into the persistent Chroma collection via
`KnowledgeRetriever.add_documents()`, printing an ingestion summary.

## Running the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now available at `http://localhost:8000`. Interactive docs at
`http://localhost:8000/docs`.

## Example API request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
        "session_id": "caller-123",
        "message": "My child has had a fever for two days, what should I do?"
      }'
```

## Example API response

```json
{
  "message": "A fever lasting two days in a child can be a sign of malaria or another infection. Please take your child to the nearest clinic today for a test, especially since fever this long needs proper diagnosis.",
  "intent": "symptom_check",
  "stage": "providing_answer",
  "risk_level": "medium",
  "continue_conversation": true,
  "confidence": 0.87,
  "sources": ["malaria_symptoms_001"]
}
```

## Future improvements

- Replace the in-memory `ConversationStore` with a durable backend (Redis
  or Postgres) so sessions survive process restarts and scale across
  multiple API instances.
- Promote the inlined dialogue branching in `Orchestrator._decide_next_step`
  into a standalone `DialogueManager` module as conversational complexity
  grows.
- Add authentication/rate limiting at the API layer for production
  deployment.
- Add multi-turn evaluation and automated regression tests against a
  fixed set of transcripts.
- Support streaming responses for lower perceived latency during voice
  calls.
