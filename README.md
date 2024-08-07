# GraphRAG


## Overview
Understanding and replication of GraphRAG with an extention to the multimodal space - Specifically incoporating visual images of recognized entites in news articles. 

## Step 1 - Locally hosted LLM  
In the orignal GraphRag, OpenAI LLM Models are used that require payments (Although it gives accurate results in terms of entities extracted and descriptions of communites). 

For locally hosted LLM using Ollama: https://github.com/TheAiSingularity/graphrag-local-ollama. 

Similary, other models such as Mixtral 8x7B LLM can be used for embedding and graph formulations, served with vLLM. https://docs.mistral.ai/deployment/self-deployment/vllm/

## Setting up
Referencing the original document: https://microsoft.github.io/graphrag/posts/get_started/