---
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
license: mit
datasets:
- julep-ai/dfe-stacked_samsum
language:
- en
library_name: sentence-transformers
---

# DFE (Dialog Fact Encoder)

This is a [sentence-transformers](https://www.SBERT.net) model: It maps "dialog" & "facts" to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search.

The Dialog Fact Encoder (DFE) is an embedding model trained to capture semantic relevance between conversational dialog turns and factual statements. It builds upon the BGE embedding model by adding a merge layer that transforms the embeddings based on whether the input text is a "dialog" or "fact".

Specifically, dialog inputs pass through additional dense layers to project the embeddings into a space optimized for comparing dialog turns. Similarly, factual inputs pass through separate dense layers to transform the embeddings for relevance matching against dialogs.

This allows DFE to embed dialogs and facts into a joint relevance space without needing to generate explicit search queries. DFE enables low-latency approximate matching of relevant facts to dialog turns, while avoiding the high computational cost of query generation models.

The model was trained using a triplet loss to pull dialog embeddings closer to relevant fact embeddings, while pushing non-relevant pairs further apart. This helps the model learn the nuanced semantics needed to assess relevance between dialogs and facts.

DFE provides an efficient way to embed variable conversational dialog into a relevance space with factual statements. This enables real-time semantic search over knowledge without expensive query generation.

> DFE is permissively licensed under the MIT license.

## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer

dialog = """
Diwank: Hey, what are we eating for dinner today?  
Ishita: Already? I thought we just ate lol  
Diwank: Yeah, some of us work hard and get hungy  
Ishita: Okay, what do you want to eat then?  
Diwank: I want to eat out but I am thinking of something light.  
""".strip()

facts = [
  "Diwank likes Sushi.",
  "Ishita does not like unnecessarily-pricey places restaurants",
  "Diwank likes cooking.",
  "Ishita is terrible at cooking.",
  "Diwank loves to eat Ishita's head.",
]

model = SentenceTransformer("julep-ai/dfe-base-en")
dialog_embeddings = model.encode({"dialog": dialog})
fact_embeddings = model.encode([{"fact": fact} for fact in facts])
```

## Background

So what's _Dialog-Fact Encoder_?

It is a new model trained by the Julep AI team to match (possibly) relevant facts to a dialog. It is an embedding model which means that it takes text as an input and outputs an embedding vector (a list of float numbers). In DFE's case, it takes an extra parameter "type" which can be either "dialog" or "fact".

A regular embedding model like openai's text-embedding-ada-2 does not distinguish between different types and gives vectors that can then be used for calculating similarity. These models are great for building search because comparing vectors is relatively cheap so for a database (of say product descriptions), you can compute vectors for every row beforehand and then when a query comes in (like: "Winter coats for women"), calculate the query's embeddings and find items using vector similarity.

Unfortunately, this does not work for dialog because conversational statements and turns within a dialog are typically not in the format of a "query". Take this case for example:

**Database**:
1. Diwank likes Sushi.
2. Ishita does not like unnecessarily-pricey places restaurants
3. Diwank likes cooking.
4. Ishita is terrible at cooking.
5. Diwank loves to eat Ishita's head.

**Dialog**:
> Diwank: Hey, what are we eating for dinner today?  
> Ishita: Already? I thought we just ate lol  
> Diwank: Yeah, some of us work hard and get hungy  
> Ishita: Okay, what do you want to eat then?  
> Diwank: I want to eat out but I am thinking of something light.  

Now, a text/vector/hybrid search would probably match all 5 facts to this conversation but, as you can see, only facts 1 and 2 are relevant. The only way to get the correct fact, right now, is to ask an LLM like gpt-3.5 to "generate a query" for querying the database and then using that for similarity. Unfortunately, there are three big problems with that:
- It adds latency and cost.
- You have to figure out "when" to run this process and retrieve (which is hard).
- The prompt for generating the query will have to be customized for every use case because the prompt has to "know" what is "query-able". So for example, in this case above, we would have to specify that you can write a query to search preferences of Diwank and Ishita from the database.

Here's where DFE comes in. The insight here is that embeddings for a dialog have meaningful information to distinguish whether a fact is relevant or not (that is exactly how we can tell that Fact 1 and 2 are relevant and others are not in the example above because we can "see" the meaning of the dialog). Normal embedding models are only interested in "overall similarity" and they'd miss this nuance, especially for details that were NOT present in the dialog directly (for example, fact 1 mentions Sushi whereas no food items are specifically mentioned in the dialog).

So, if this information is already there in theory, how can we learn to connect embeddings of "facts" and "dialogues" based on relevance? DFE is trained to do exactly that. DFE is about learning this "relevance" transformation of a dialog so it matches similar facts.

DFE is a built upon BGE (currently the best state-of-the-art model for embeddings). It has the base embeddings from the original BGE model and added dense layers. The base BGE model is actually frozen and left completely unchanged because it already knows how to turn a passage into an embedding very well. We add the new layers to learn how to "transform" the input depending on what kind of passage it is (a dialog or a fact) in a way that "relevant" stuff is closer together in the embedding space.

This solves all of the three problems from the "query generation" method from earlier. Instead of generating a query using an LLM, you can store facts with their DFE embeddings in the database beforehand and then just embed the dialog using DFE and match. Since this operation is so much faster, you can basically do this on every turn without much hassle.

The "query generation" method is still far superior in quality but is too prohibitive (costly + slow) in normal circumstances and DFE solves that. :)

## Technical details

It inherits the base BERT model and pooling layer from BGE to generate 768-dimensional embeddings for input text.

DFE then adds an Asymmetric projection layer with separate dense layers for "dialog" and "fact" inputs:

1. Dialog inputs pass through 2x1536D tanh layers, a dropout layer, and another 1536D tanh layer before projecting back to 768 dimensions.
2. Fact inputs pass through similar 1536D tanh layers with dropout before projecting back to 768D.
3. This asymmetric architecture allows specialization of the embeddings for relevance matching between dialogs and facts.

DFE is trained with a triplet loss using the TripletDistanceMetric.EUCLIDEAN distance function and a margin of 5. It pulls dialog embeddings closer to positively matched fact embeddings, while pushing non-relevant pairs beyond the margin.

This approach teaches DFE to transform dialog and fact embeddings into a joint relevance space optimized for low-latency semantic matching. The specialized projections allow fast approximation of relevant facts for conversational dialog turns.

## Dataset

The model was trained on a custom dataset [julep-ai/dfe-stacked_samsum](https://huggingface.co/datasets/julep-ai/dfe-stacked_samsum) that we created from [stacked-summaries/stacked-samsum-1024](https://huggingface.co/datasets/stacked-summaries/stacked-samsum-1024) by:
1. Extracting summaries for corresponding dialogs to emulate "facts"
2. Then truncating the dialogs to emulate "missing information"
3. And then augmenting the dialogs using LLMs to emulate "additional information"

## Training
The model was trained with the parameters:

**Loss**:

`sentence_transformers.losses.TripletLoss.TripletLoss` with parameters:
  ```
  {'distance_metric': 'TripletDistanceMetric.EUCLIDEAN', 'triplet_margin': 5}
  ```

Parameters of the fit()-Method:
```
{
    "epochs": 12,
    "evaluation_steps": 0,
    "evaluator": "sentence_transformers.evaluation.TripletEvaluator.TripletEvaluator",
    "max_grad_norm": 1,
    "optimizer_class": "<class 'lion_pytorch.lion_pytorch.Lion'>",
    "optimizer_params": {
        "lr": 0.0001,
        "weight_decay": 0.01
    },
    "scheduler": "WarmupCosine",
    "steps_per_epoch": null,
    "warmup_steps": 100,
    "weight_decay": 0.01
}
```

## Evaluation Results

<!--- Describe how your model was evaluated -->

TBD

## Full Model Architecture
```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': True}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
  (2): Asym(
    (dialog-0): Dense({'in_features': 768, 'out_features': 1536, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
    (dialog-1): Dense({'in_features': 1536, 'out_features': 1536, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
    (dialog-2): Dropout(
        (dropout_layer): Dropout(p=0.1, inplace=False)
    )
    (dialog-3): Dense({'in_features': 1536, 'out_features': 1536, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
    (dialog-4): Dense({'in_features': 1536, 'out_features': 768, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
    (fact-0): Dense({'in_features': 768, 'out_features': 1536, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
    (fact-1): Dense({'in_features': 1536, 'out_features': 1536, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
    (fact-2): Dropout(
        (dropout_layer): Dropout(p=0.1, inplace=False)
    )
    (fact-3): Dense({'in_features': 1536, 'out_features': 1536, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
    (fact-4): Dense({'in_features': 1536, 'out_features': 768, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
  )
)
```

## Citing & Authors

```
Diwank Singh Tomer, Julep AI Inc. Dialog Fact Encoder (DFE). https://julep.ai (2023).
```