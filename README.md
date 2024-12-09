# Haystack: Unstructured + Qdrant + Ollama

This repo provides a docker configuration to run qdrant vectore database, index files from ***data/*** folder and and answer a question hard-coded in ***scripts/retrieval*** as a RAG.

This repo is heavily inspired by Group 1.2. It lets you create sample answers from the ***dataset.csv*** which can be evaluated using a notebook in Google Colab.

## Steps to reproduce


- Run `docker-compose up --build -d`
- Download required ollama model via `docker exec <container_id> ollama pull llama3.2:1b` (get container id by running `docker ps`)
- Change HOW_MANY in .env, for  desired sample size
- Use the main.ipynb to create sample answers and questions for evaluating in final.csv
  - Alternative `!pip install -r requirements.txt` and `python main.py`
- Open [Colab-Eval](https://colab.research.google.com/drive/13wrtX95EBizfnHE3PvGRENG_FolDwYep?usp=sharing) and upload final.csv
- Run the notebook to get the results in the last cell