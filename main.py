# %%

# %%
import dotenv
import os

dotenv.load_dotenv(".env", override=True)
print(os.getenv("UNSTRUCTURED_API_KEY"))
print(os.getenv("HOW_MANY"))

# %% [markdown]
# # Generating the answers

# %% [markdown]
# ## Loading questions

# %%
import pandas as pd
import os

df = pd.read_csv("dataset.csv", delimiter=",")
questions = df["question"].tolist()[: int(os.getenv("HOW_MANY"))]
ground_truths = df["correct"].tolist()[: int(os.getenv("HOW_MANY"))]

indeces = df["id"].tolist()[: int(os.getenv("HOW_MANY"))]

filenames = os.listdir(os.getenv("LOCAL_FILE_INPUT_DIR"))

# filepaths = [os.path.join(os.getenv("LOCAL_FILE_INPUT_DIR"), filename) for filename in filenames if not filename.startswith(".")]
filepaths = [
    os.path.join(os.getenv("LOCAL_FILE_INPUT_DIR"), filename)
    for filename in filenames
    if os.path.splitext(filename)[0] in indeces
]


# %% [markdown]
# ## Indexing

# %%
from haystack_integrations.components.converters.unstructured import (
    UnstructuredFileConverter,
)
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedDocumentEmbedder,
)
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.components.embedders.fastembed import (
    FastembedTextEmbedder,
    FastembedSparseTextEmbedder,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack import Document, Pipeline

qdrant_db_sparse = QdrantDocumentStore(
    url="http://localhost:6333",  # Adjust this if your Qdrant is hosted elsewhere
    index="haystack_index",  # Use the name of your existing Qdrant index
    recreate_index=True,  # Ensure we don't overwrite the existing database
    embedding_dim=384,
    return_embedding=True,  # Return embeddings from Qdrant
    use_sparse_embeddings=True,
    sparse_idf=True,
)

doc_embedder = FastembedDocumentEmbedder(model="BAAI/bge-small-en-v1.5")
doc_embedder.warm_up()
sparse_doc_embedder = FastembedSparseDocumentEmbedder(
    model="Qdrant/bm42-all-minilm-l6-v2-attentions"
)
sparse_doc_embedder.warm_up()


hybrid_indexing = Pipeline()
hybrid_indexing.add_component(
    "converter",
    UnstructuredFileConverter(
        api_url="https://api.unstructuredapp.io/general/v0/general",
        document_creation_mode="one-doc-per-element",
    ),
)
hybrid_indexing.add_component(
    "sparse_doc_embedder",
    FastembedSparseDocumentEmbedder(model="Qdrant/bm42-all-minilm-l6-v2-attentions"),
)
hybrid_indexing.add_component(
    "dense_doc_embedder", FastembedDocumentEmbedder(model="BAAI/bge-small-en-v1.5")
)
hybrid_indexing.add_component(
    "writer",
    DocumentWriter(document_store=qdrant_db_sparse, policy=DuplicatePolicy.OVERWRITE),
)

hybrid_indexing.connect("converter", "sparse_doc_embedder")
hybrid_indexing.connect("sparse_doc_embedder", "dense_doc_embedder")
hybrid_indexing.connect("dense_doc_embedder", "writer")

hybrid_indexing.run({"paths": filepaths})

# %% [markdown]
# ## Retreiver

# %%
from haystack_integrations.components.converters.unstructured import (
    UnstructuredFileConverter,
)
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedDocumentEmbedder,
)
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.components.embedders.fastembed import (
    FastembedTextEmbedder,
    FastembedSparseTextEmbedder,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack import Document, Pipeline
from haystack_integrations.components.generators.ollama import OllamaGenerator
import csv

qdrant_db_sparse = QdrantDocumentStore(
    url="http://localhost:6333",  # Adjust this if your Qdrant is hosted elsewhere
    index="haystack_index",  # Use the name of your existing Qdrant index
    recreate_index=False,  # Ensure we don't overwrite the existing database
    return_embedding=True,  # Return embeddings from Qdrant
    use_sparse_embeddings=True,
    sparse_idf=True,
    embedding_dim=384,
)

system_prompt = "You are a helpful assistant. Answer the question based on the provided information. Answer concisely and informatively. If you don't know the answer, say so."
generation_kwargs = {
    "seed": 42,
    # "temperature": 0.8,
    # "repeat_penalty": 1.1,
    # "num_predict": 128, # max number of tokens to generate
    # "top_k": 50, # top-k sampling
    # "top_p": 0.9, # top-p sampling
    # "min_p": 0.0 # filter out token with probability less than this
}
generator = OllamaGenerator(
    model="llama3.2:1b",
    url="http://localhost:11434",
    system_prompt=system_prompt,
    generation_kwargs=generation_kwargs,
)

doc_embedder = FastembedDocumentEmbedder(model="BAAI/bge-small-en-v1.5")
doc_embedder.warm_up()
sparse_doc_embedder = FastembedSparseDocumentEmbedder(
    model="Qdrant/bm42-all-minilm-l6-v2-attentions"
)
sparse_doc_embedder.warm_up()

hybrid_query = Pipeline()
hybrid_query.add_component(
    "sparse_text_embedder",
    FastembedSparseTextEmbedder(model="Qdrant/bm42-all-minilm-l6-v2-attentions"),
)
hybrid_query.add_component(
    "dense_text_embedder",
    FastembedTextEmbedder(
        model="BAAI/bge-small-en-v1.5",
        prefix="Represent this sentence for searching relevant passages: ",
    ),
)
hybrid_query.add_component(
    "retriever", QdrantHybridRetriever(document_store=qdrant_db_sparse, top_k=5)
)

hybrid_query.connect(
    "sparse_text_embedder.sparse_embedding", "retriever.query_sparse_embedding"
)
hybrid_query.connect("dense_text_embedder.embedding", "retriever.query_embedding")

prompt_template = """
{sources_text}

{question}
"""
with open("final.csv", "w") as f:
    writer = csv.writer(f, delimiter="|")
    writer.writerow(["question", "ground_truth", "gen_answer"])
    for i, question in enumerate(questions):
        results = hybrid_query.run(
            {
                "dense_text_embedder": {"text": question},
                "sparse_text_embedder": {"text": question},
            }
        )
        sources = [result.content for result in results["retriever"]["documents"]]
        prompt = prompt_template.format(
            sources_text="\n\n".join(sources), question=question
        )

        gen_answer = generator.run(prompt, generation_kwargs=generation_kwargs)[
            "replies"
        ]
        writer.writerow([question, ground_truths[i], gen_answer[0]])
