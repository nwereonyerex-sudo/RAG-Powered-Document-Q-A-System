# SparseRoute: smaller indexes for dense retrieval

## Abstract

We introduce SparseRoute, a retrieval method that cuts index size by 35 percent on the BEIR benchmark while holding nDCG within 0.01 of the full index.
The method keeps a centroid sketch of each chunk and drops vectors that sit inside an existing centroid ball.
Experiments used the same MiniLM embeddings for every condition.

## Method

SparseRoute builds the sketch in one pass over the chunk embeddings.
It does not retrain the embedding model.
On the legislation subset, mean query latency fell from 40 milliseconds to 27 milliseconds.
