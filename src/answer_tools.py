from .models import MinimalAnswer , MinimalSource
from transformers import AutoModelForForCausalLM, AutoTokenizer, AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B'), AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
from .io import _read_document
from rag_tools  import searcher

def answer_generator(query: str, retrieved_sources: List[MinimalSource],k: int) -> List[MinimalAnswer]:
    

    

