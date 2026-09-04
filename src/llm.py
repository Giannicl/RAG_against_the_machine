"""Local causal language model wrapper for grounded answer generation."""

import re
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .errors import InputError

_MODEL_NAME = "Qwen/Qwen3-0.6B"
_MAX_NEW_TOKENS = 192
_BATCH_SIZE = 4
_DO_SAMPLE = False
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def _thinking_stripper(completion: str) -> str:
    """Drop any <think>...</think> span a reasoning model emitted.

    Args:
        completion: Decoded model completion.

    Returns:
        The completion without thinking spans, whitespace-stripped.
    """
    return _THINK_PATTERN.sub("", completion).strip()


class LLM:
    """Chat-tuned causal language model answering prompts in batches."""

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        """Load the tokenizer and model weights.

        Args:
            model_name: Hugging Face name of the model to load.

        Raises:
            InputError: The model cannot be downloaded or loaded.
        """
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(model_name)
        except (OSError, ValueError) as error:
            raise InputError(
                f"cannot load model {model_name}: {error}"
            ) from error
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model.eval()

    def _chat_prompt_builder(self, prompt: str) -> str:
        """Wrap a prompt in the model's chat template, thinking off.

        Args:
            prompt: The user-turn text.

        Returns:
            The templated prompt string.
        """
        chat_messages = [{"role": "user", "content": prompt}]
        try:
            return str(
                self._tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except TypeError:
            return str(
                self._tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

    def _batch_generator(self, chat_prompts: List[str]) -> List[str]:
        """Generate one completion per templated prompt, in one batch.

        Args:
            chat_prompts: Templated prompts, at most _BATCH_SIZE long.

        Returns:
            One decoded completion per prompt, same order.
        """
        model_inputs = self._tokenizer(
            chat_prompts, return_tensors="pt", padding=True
        ).to(self._model.device)
        with torch.inference_mode():
            generate_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=_DO_SAMPLE,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        completion_ids = generate_ids[
            :, model_inputs["input_ids"].shape[1]:
        ]
        completions = self._tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        return [
            _thinking_stripper(completion) for completion in completions
        ]

    def generate(self, prompts: List[str]) -> List[str]:
        """Answer every prompt, batched with a progress bar.

        Args:
            prompts: Prompts to answer, one per question.

        Returns:
            One completion per prompt, in the same order.
        """
        chat_prompts = [
            self._chat_prompt_builder(prompt) for prompt in prompts
        ]
        completions: List[str] = []
        prompt_batches = [
            chat_prompts[start:start + _BATCH_SIZE]
            for start in range(0, len(chat_prompts), _BATCH_SIZE)
        ]
        for prompt_batch in tqdm(prompt_batches, desc="generating answers"):
            completions.extend(self._batch_generator(prompt_batch))
        return completions
