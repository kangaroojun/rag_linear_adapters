# Linear Adapters

## Overview

Linear Adapters were explored in my internship as part of improving query embeddings for retrieval. It involves training a single layer neural network with sample queries and their golden chunks such that the query embeddings will be more closely aligned with embeddings of golden chunks. More details can be found in the [documentation](./docs/documentation.md).

The codebase consists of 3 classes, `LinearAdapter`, `LinearAdapterTrainer` and `LinearAdapterFinetuneEngine`. When implementing Linear Adapters, you would just need to work with the former 2 classes: `LinearAdapter` is a Haystack component that can be used in place of Dense Embedder object for embedding queries, while `LinearAdapterTrainer` takes in a `myDataset` object for training, which is from the `DatasetGenerator` class.

## Usage Examples

Here is a basic example of how to use this:

```python
# Get train dataset
train_dataset = myDataset.from_json('data/cleaned_train_multi.json')

# Get document store
milvus_document_store = MilvusDocumentStore(...)

# Call trainer object
trainer = LinearAdapterTrainer(embedding_model_name="azure", milvus_document_store=milvus_document_store)

# Call train method
trainer.train_linear_adapter(dataset=train_dataset, epochs=5, model_output_path="./models/linear_adapter_multi")

# Run adapter (used in place of dense embedder)
adapter = LinearAdapter(model_name='azure', adapter_path='./models/linear_adapter_multi')
adapter.warm_up()
embeddings_linear_adapter = adapter.run("What is the capital of France?")
print(embeddings_linear_adapter['embedding']) 
```

```
>> [-0.05556660145521164, 0.04077581316232681, ...]
```

See our runs in our [notebook](./notebook/linear_adapters.ipynb).

## Developer Notes

**Training Dataset**: To obtain a training dataset, take a look at the dataset generation guide.

**LinearAdapterFinetuneEngine:** Inherits from LlamaIndex's `EmbeddingAdapterFinetuneEngine`, ensuring that the chunks are not re-embedded each time during training, but instead retrieving their embeddings from the Milvus database. This saves cost and also improves performance in the event that we have implemented methodologies such as late semantic chunking, whereby the embedding values of each chunk would factor in the entire document.