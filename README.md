# Haystack: Unstructured + Qdrant + Ollama

This repo is heavily inspired by Group 1.2. It lets you create sample answers from the ***dataset.csv*** which can be evaluated using a notebook in Google Colab.

For that it creates a Qdrant databse which is loaded with documents from Unstructured and which are used as a RAG for answering question using Ollama to then Evaluate these Questions uses Prometheus

## Steps to reproduce

- Run in the directory `python -m venv .venv`
- Run `source .venv/bin/activate`
- Run `pip install -r requirements`
- Run `docker-compose up --build -d`
- Download required ollama model via `docker exec <container_id> ollama pull llama3.2:1b` (get container id by running `docker ps`)
- Change HOW_MANY in .env, for  desired sample size
- Use the main.ipynb to create sample answers and questions for evaluating in final.csv
  - Alternative `python main.py`
- Open [Colab-Eval](https://colab.research.google.com/drive/13wrtX95EBizfnHE3PvGRENG_FolDwYep?usp=sharing) and upload final.csv
- Run the notebook to get the evaluation in the end

## Other
The Evaluation tool ist also availibe as a notebook, but depends on CUDA GPU, thats why we run it on COLAB

Tested on python 3.12.3 (check with `python --version`)