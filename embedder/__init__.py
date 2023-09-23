#!/usr/bin/env python3

from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from FlagEmbedding import FlagModel
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


###########
## Model ##
###########

model_names: list[str] = [
    "BAAI/bge-small-en-v1.5",
    "julep-ai/dfe-base-en",
]

default_model: str = model_names[0]


@lru_cache(maxsize=1)
def get_model(model_name: str):
    query_instruction_for_retrieval = (
        "Represent this sentence for searching relevant passages:"
    )

    if model_name.startswith("BAAI/"):
        model = FlagModel(
            model_name,
            query_instruction_for_retrieval=query_instruction_for_retrieval,
            pooling_method="cls",
            normalize_embeddings=True,
            use_fp16=True,
        )

    else:
        model = SentenceTransformer(model_name)

    return model


#############
## FastAPI ##
#############

app = FastAPI()

@app.on_event("startup")
async def lifespan():
    # Load the ML models
    for name in model_names:
        get_model(name)


###########
## Types ##
###########

class Request(BaseModel):
    instances: list[str]
    parameters: dict[str, Any] = {
        "model_name": default_model,
        "query_type": "passage",
    }


############
## Routes ##
############

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed")
def embed(request: Request) -> dict[str, list[list[float]]]:

    # Constants
    instances = request.instances
    parameters = request.parameters
    model_name = parameters.get("model_name", default_model)
    query_type = parameters.get("query_type", "passage")

    # Get the model
    model = get_model(model_name)
    encoder = model.encode_queries if query_type == "query" else model.encode

    # Wrap instances if using the custom model
    if "dfe" in model_name:
        assert query_type in ["dialog", "fact"], \
            '`query_type` must be "dialog" or "fact"'
        
        instances = [{query_type: instance} for instance in instances]

    # Embed the instances
    predictions = encoder(instances)

    return dict(predictions=predictions.tolist())
