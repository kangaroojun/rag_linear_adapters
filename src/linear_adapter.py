####################
# Required Modules #
####################

# Generic/Built-in
import os
from typing import List, Optional, Literal, Any, Tuple

# Libs
import torch
from haystack import component
from haystack import Document
from milvus_haystack import MilvusDocumentStore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.adapter import LinearAdapterEmbeddingModel
from llama_index.finetuning import (
    EmbeddingAdapterFinetuneEngine,
    EmbeddingQAFinetuneDataset
)
from torch import Tensor
from torch.utils.data import DataLoader

# Custom
from .utils import myDataset

##############################
# Retrieval - Linear Adapter #
##############################

@component
class LinearAdapter:
    """ A component that uses a pre-trained linear adapter to embed text queries. The adapter is trained on a myDataset
    object generated from Dataset Generation component. The adapter is trained on a given dataset of question-answer pairs
    and then used to embed queries. You may use this component as a replacement for the query embedder in the Retriever.

    Attributes:
        model_name (str): The name of the model to use.
        adapter_path (str): The path to the adapter.
        prefix (str): The prefix to add to the input text.
        suffix (str): The suffix to add to the input text.
        _base_embed_model (AzureOpenAIEmbedding): The base embedding model instance.
        _embed_model (LinearAdapterEmbeddingModel): The linear adapter embedding model instance.
    """

    def __init__(self, model_name: Literal['azure', 'jinaai'] , adapter_path: str) -> None:
        """Initializes the LinearAdapter component.
        
        Args:
            model_name (str): The name of the model to use.
            adapter_path (str): The path to the adapter.

        Raises:
            NotImplementedError: If the model name is not recognized.
        """
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.prefix = ""
        self.suffix = ""

    def warm_up(self) -> None:
        """Warm up the model by loading it into memory."""
        if self.model_name == 'azure':
            self._base_embed_model = AzureOpenAIEmbedding(
                engine=os.getenv("AZURE_OPENAI_EMBEDDER"),
                model=os.getenv("AZURE_OPENAI_EMBEDDER"),        
                api_version=os.getenv("AZURE_OPENAI_VERSION"),
                api_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
        else:
            raise NotImplementedError(f"Model name '{self.model_name}' not recognized.")
        self._embed_model = LinearAdapterEmbeddingModel(self._base_embed_model, "./models/linear_adapter")

    @property
    def base_embed_model(self) -> AzureOpenAIEmbedding:
        """Returns the base embedding model instance"""
        return self._base_embed_model
    
    @property
    def embed_model(self) -> LinearAdapterEmbeddingModel:
        """Returns the linear adapter embedding model instance"""
        return self._embed_model
    
    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """Runs the LinearAdapter component on the given text.

        Args:
            text (str): The query to embed.

        Returns:
            Dict[str, List[float]]: The embedding of the query.
        """
        if not isinstance(text, str):
            # Check if input is a list and all elements are instances of Document
            if isinstance(text, list) and all(isinstance(elem, Document) for elem in text):
                error_message = "Input must be a string. Use AzureOpenAIDocumentEmbedder for a list of Documents."
            else:
                error_message = "Input must be a string."
            raise TypeError(error_message)

        # Preprocess the text by adding prefixes/suffixes
        # finally, replace newlines as recommended by OpenAI docs
        processed_text = f"{self.prefix}{text}{self.suffix}".replace("\n", " ")
        
        query_embedding = self.embed_model._get_query_embedding(processed_text)
    
        return {
            "embedding": query_embedding,
        }

class LinearAdapterTrainer:
    """A class that encapsulates the training and evaluation of a linear adapter for a given dataset.

    Attributes:
        embedding_model_name (str): The name of the embedding model to use.
        llm_model_name (str): The name of the language model to use.
        _embed_model (AzureOpenAIEmbedding): The Azure OpenAI embedding model instance.
        _document_store (MilvusDocumentStore): The Milvus Document Store instance.
        _llm (AzureOpenAI): The Azure OpenAI language model instance.
    """

    def __init__(
            self, 
            embedding_model_name: Literal['azure', 'jinaai'],
        ) -> None:
        """Initialization logic involving loading the Azure API credentials from environment variables and
        initializing the required components.

        Args:
            model_name (str): The name of the model to use.

        Raises:
            NotImplementedError: If the model name is not recognized.
        """

        self.model_name = embedding_model_name

        if self.model_name == 'azure':
            self._embed_model = AzureOpenAIEmbedding(
                engine=os.getenv("AZURE_OPENAI_EMBEDDER"),
                model=os.getenv("AZURE_OPENAI_EMBEDDER"),        
                api_version=os.getenv("AZURE_OPENAI_VERSION"),
                api_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
        else:
            raise NotImplementedError(f"Model name '{self.model_name}' not recognized.")

        self._document_store = MilvusDocumentStore(
            collection_name="test_indexing_pipeline",
            collection_description="testindexingpipeline",
            connection_args={
                "host": os.getenv("MILVUS_HOST", "localhost"),
                "port": os.getenv("MILVUS_PORT", "19530"),
                "user": "",
                "password": "",
                "secure": False,
            },
        )

    @property
    def embed_model(self) -> AzureOpenAIEmbedding:
        """Returns the Azure OpenAI embedding model instance"""
        return self._embed_model
    
    @property
    def document_store(self) -> MilvusDocumentStore:
        """Returns the Milvus Document Store instance"""
        return self._document_store

    def train_linear_adapter(
            self,
            dataset: myDataset, 
            embedder_model: Optional[Any] = None, 
            epochs: int = 3, 
            model_output_path: str = "./models/linear_adapter",
            verbose: bool = False
        ) -> LinearAdapterEmbeddingModel:
        """
        Finetunes a linear adapter on a given dataset of question-answer pairs.

        Args:
            `dataset` (EmbeddingQAFinetuneDataset): The dataset to be used for finetuning.
            `embedder_model` (AzureOpenAIEmbedding): The embedding model to be used for finetuning.
            `epochs` (int): The number of epochs to train the adapter for (default: 3).
            `model_output_path` (str): The path to save the finetuned adapter model (default: "./models/linear_adapter").
            `verbose` (bool): Whether to print progress information (default: False).    
        
        Returns:
            None
        """
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path)
        if embedder_model is None:
            embedder_model = self.embed_model
        finetune_engine = LinearAdapterFinetuneEngine(
            dataset=dataset,
            embed_model=self.embedder_model if embedder_model is None else embedder_model,
            document_store=self.document_store,
            epochs=epochs,
            model_output_path=model_output_path,
            verbose=verbose
        )
        finetune_engine.finetune()
        return finetune_engine.get_finetuned_model()
    
class LinearAdapterFinetuneEngine(EmbeddingAdapterFinetuneEngine):
    """A class that encapsulates the training and evaluation of a linear adapter for a given dataset. This class is
    adapted to use embeddings from Milvus instead of re-embedding each chunk.

    Attributes:
        dataset (myDataset): The dataset to be used for finetuning.
        embed_model (AzureOpenAIEmbedding): The Azure OpenAI embedding model instance.
        document_store (MilvusDocumentStore): The Milvus Document Store instance.
        batch_size (int): The batch size to use for training.
        epochs (int): The number of epochs to train the adapter for.
        adapter_model (Optional[LinearAdapterEmbeddingModel]): The adapter model to be finetuned.
        dim (int): The dimension of the adapter model.
        device: The device to use for training.
        model_output_path (str): The path to save the finetuned adapter model.
        model_checkpoint_path (str): The path to save the model checkpoints.
        checkpoint_save_steps (int): The number of steps after which to save a checkpoint.
        verbose (bool): Whether to print progress information.
        bias (bool): Whether to use a bias term in the linear adapter.

    """
    def __init__(
        self,
        dataset: myDataset,
        embed_model: AzureOpenAIEmbedding,
        document_store: MilvusDocumentStore,
        batch_size: int = 10,
        epochs: int = 1,
        adapter_model=None,
        dim: int = None,
        device = None,
        model_output_path: str = "model_output",
        model_checkpoint_path: str = None,
        checkpoint_save_steps: int = 100,
        verbose: bool = False,
        bias: bool = False,
        **train_kwargs
    ):
        super().__init__(
            dataset=dataset,
            embed_model=embed_model,
            batch_size=batch_size,
            epochs=epochs,
            adapter_model=adapter_model,
            dim=dim,
            device=device,
            model_output_path=model_output_path,
            model_checkpoint_path=model_checkpoint_path,
            checkpoint_save_steps=checkpoint_save_steps,
            verbose=verbose,
            bias=bias,
            **train_kwargs
        )
        self.document_store = document_store

    def smart_batching_collate(self, batch: Tuple[str, str]) -> Tuple[Tensor, Tensor]:
        """Collate function for DataLoader, adapted to get embeddings from Milvus instead of re-embedding each chunk.

        Args:
            batch (Tuple[str, str]): A list of tuples containing the query and chunk_id.

        Returns:
            Tuple[Tensor, Tensor]: The query and text embeddings to be used for finetuning.
        """
        query_embeddings: List[Tensor] = []
        text_embeddings: List[Tensor] = []
        for query, chunk_id in batch:
            query_embedding = self.embed_model.get_query_embedding(query)
            text_embedding = self.document_store.col.query(
                expr=f"id == '{chunk_id}'", 
                output_fields=["vector"]
            )[0]['vector']
            query_embeddings.append(torch.tensor(query_embedding))
            text_embeddings.append(torch.tensor(text_embedding))
        query_embeddings_t = torch.stack(query_embeddings)
        text_embeddings_t = torch.stack(text_embeddings)
        return query_embeddings_t, text_embeddings_t

    def _get_data_loader(self, dataset: EmbeddingQAFinetuneDataset) -> DataLoader:
        """Get data loader for the finetuning dataset. Adapted to get chunk_ids instead of chunk itself, so as to access 
        the embeddings from Milvus.

        Args:
            dataset (EmbeddingQAFinetuneDataset): The dataset to be used for finetuning.

        Returns:
            DataLoader: The DataLoader instance for the finetuning dataset.
        """
        examples: Any = []
        for query_id, query in dataset.queries.items():
            chunk_ids = dataset.relevant_docs[query_id]
            for chunk_id in chunk_ids:
                examples.append((query, chunk_id))
        data_loader = DataLoader(examples, batch_size=self.batch_size)
        data_loader.collate_fn = self.smart_batching_collate
        return data_loader