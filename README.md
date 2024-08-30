# local agent project

We gonn' use mem0 and llamaindex/ollama to build a local agent.

## Step 1: Start a qdrant instance

```bash
docker pull qdrant/qdrant
```

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
