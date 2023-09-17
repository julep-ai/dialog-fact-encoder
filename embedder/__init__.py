#!/usr/bin/env python3

from functools import lru_cache

from fastapi import FastAPI
from FlagEmbedding import FlagModel
from pydantic import BaseModel


###########
## Model ##
###########

@lru_cache(maxsize=1)
def get_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    query_instruction_for_retrieval = (
        "Represent this sentence for searching relevant passages:"
    )

    model = FlagModel(
        model_name,
        query_instruction_for_retrieval=query_instruction_for_retrieval,
        pooling_method="cls",
        normalize_embeddings=True,
        use_fp16=True,
    )

    return model


#############
## FastAPI ##
#############

app = FastAPI()

@app.on_event("startup")
async def lifespan():
    # Load the ML model
    get_model()


###########
## Types ##
###########

class Request(BaseModel):
    instances: list[str]


############
## Routes ##
############

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed")
def embed(request: Request) -> dict[str, list[list[float]]]:

    model = get_model()
    predictions = model.encode(
        request.instances,
        max_length=512,
    )

    return dict(predictions=predictions.tolist())
