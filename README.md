# RAG-Fusion: The Next Frontier of Search Technology

## Overview

The code accompanying this README is an implementation of RAG-Fusion, a search methodology that aims to bridge the gap between traditional search paradigms and the multifaceted dimensions of human queries. Inspired by the capabilities of Retrieval Augmented Generation (RAG), this project goes a step further by employing multiple query generation and Reciprocal Rank Fusion to re-rank search results. The overarching goal is to move closer to unearthing that elusive 90% of transformative knowledge that often remains hidden behind top search results.

## Context

The rise of Retrieval Augmented Generation (RAG) has shifted paradigms in the AI and search space by fusing the power of vector search with generative models. However, the technology is not without its constraints. Traditional search methods often fall short when it comes to understanding the nuances and complexities of human queries.

That's where RAG-Fusion comes in. Detailed in the article, "The Next Frontier of Search: Retrieval Augmented Generation meets Reciprocal Rank Fusion and Generated Queries," this methodology aims to tackle these challenges head-on.

## What The Code Does

1. **Query Generation**: The system starts by generating multiple queries from a user's initial query using OpenAI's GPT model.
  
2. **Vector Search**: Conducts vector-based searches on each of the generated queries to retrieve relevant documents from a predefined set.

3. **Reciprocal Rank Fusion**: Applies the Reciprocal Rank Fusion algorithm to re-rank the documents based on their relevance across multiple queries.

4. **Output Generation**: Produces a final output consisting of the re-ranked list of documents.

## How to Run the Code

1. Install the required dependencies (`openai`).
2. Place your OpenAI API key in the appropriate spot in the code.
3. Run the script.

## Why RAG-Fusion?

RAG-Fusion is an ongoing experiment that aims to make search smarter and more context-aware, thus helping us uncover the richer, deeper strata of information that we might not have found otherwise.