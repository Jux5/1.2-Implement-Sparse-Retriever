# Haystack: Unstructured + Qdrant + Ollama

This repo provides a docker configuration to run qdrant vectore database, index files from ***data/*** folder and and answer a question hard-coded in ***scripts/retrieval*** as a RAG.

## Steps to reproduce

- Check the validity of unstructured serverless api key in ***.env***
- Change any parameters in ***.env*** if required
- Put your data into ***data/*** folder
- (Optional) Change connection to Qdrant in ***scripts/indexing.py*** (`recreate_index=False` if you already have a sparse index)
- Run `docker compose up --build -d`
- Wait for the indexing app container to finish
- Download required ollama model via `docker exec <container_id> ollama pull <model_name>` (<model_name> = $MODEL_NAME)
- (Optional) Change ***scripts/retrieval.py*** to pass your question as an argument
- Run `python3 scripts/retrieval.py` (you would need all the dependencies for requirements.txt for this, e.g. in a container)
- Enjoy an answer to your question in stdout
- Done! 

You can make indexing part more flexible using indexing app container (***scripts/indexing.py*** specifically) as a reference.

You can include the answer-generation part to your scripts taking ***scripts/retrieval.py*** as a reference.