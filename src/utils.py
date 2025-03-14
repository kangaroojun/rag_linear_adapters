from typing import Optional, Dict
from llama_index.finetuning import EmbeddingQAFinetuneDataset

class myDataset(EmbeddingQAFinetuneDataset):
    """Inherits from LlamaIndex's EmbeddingQAFinetuneDataset class. This class is used to store the dataset generated,
    with additional attributes to store the expected answers for the questions generated. Contains queries, corpus,
    relevant_docs and expected_answers attributes to store the questions, contexts, and expected answers respectively. 
    Access any of these attributes to get the data stored in the dataset. Methods like save_json and from_json are used
    to save and load the dataset as a JSON file.

    """
    expected_answers: Optional[Dict[str, str]] = None