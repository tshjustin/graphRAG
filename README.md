# GraphRAG


## Overview
Understanding and replication of GraphRAG with an extention to the multimodal space - Specifically incoporating visual images of recognized entites in news articles. 

## Step 1 - Locally hosted LLM  
In the orignal GraphRag, OpenAI LLM Models are used that require payments (Although it gives accurate results in terms of entities extracted and descriptions of communites). 

For locally hosted LLM using Ollama: https://github.com/TheAiSingularity/graphrag-local-ollama. 

Similary, other models such as Mixtral 8x7B LLM can be used for embedding and graph formulations, served with vLLM. https://docs.mistral.ai/deployment/self-deployment/vllm/

## Setting up
Referencing the original document: https://microsoft.github.io/graphrag/posts/get_started/

```
conda create python==3.10 -n grag
conda activate mmgl
pip install graphrag
pip install deepface

mkdir -p ./ragtest/input
python -m graphrag.index --init --root ./ragtest  # add in data
python -m graphrag.index --root ./ragtest  # run after setting up LLM and embedding model
```

| Directory/File | description |
| ---- | ---- |
| entity_matching/ | Codes related to extraction of facial features and saving known entites in a JSON format|
| graphrag/ | Codes related to the building of Grapg representation |
| graphrag/index | Main code logic folder |
| graphrag/index/graph | Utilities for graph embedding process|
| graphrag/index/input | entry point for data |
| graphrag/index/input/verbs | Main logic for code - Including entity extraction, relationships, covariates and graph embeddings |
| output/../ | Result from a run of 15 sample articles |         \_______ # This 2 files are to be placed in the created ragtest./ folder 
| prompts | Containing the prompts to guide the LLM extraction | / 
| language_modelling/utils.py | utility functions |
| graph_visuals.ipynb/ | Visualization of local portion of graph from a prompt |

## Visualization 
Using the `graph_visuals.ipynb`, the results can be visualized as shown below: 
![image](https://github.com/user-attachments/assets/e60344f8-d574-4219-8dfa-97c616380b1d)




