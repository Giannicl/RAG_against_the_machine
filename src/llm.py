from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List

_MODEL_NAME = "Qwen/Qwen3-0.6B"


class LLM:
    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: List[str]) -> str:
        model_inputs = self._tokenizer(prompt, return_tensors="pt").to(
            self._model.device
        )
        generate_ids = self._model.generate(**model_inputs)
        return self._tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
