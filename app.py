"""
Minimal vLLM demo script.

This module provides a single helper, `demo`, which loads a Hugging Face
text generation model via vLLM and prints a short completion for a given
prompt. It uses vLLM's default model implementation and tuned sampling
parameters for broad compatibility and more meaningful outputs.
"""

# Try to use vLLM if available; otherwise fall back to Transformers on CPU.
try:
    # Ensure real availability by importing LLM entrypoints (fails on Windows without vllm._C)
    from vllm import LLM as _LLM, SamplingParams as _SamplingParams  # type: ignore
    USE_VLLM = True
    print("Using vLLM for fast inference.")
except (ImportError, RuntimeError):
    USE_VLLM = False
    print("vLLM not available; falling back to Transformers on CPU.")


def _render_prompt(model_name: str, prompt: str) -> str:
    """Render a chat-style prompt when the tokenizer provides a chat template.

    Falls back to a simple instruction-wrapped text prompt when no template
    is available or if anything fails during tokenizer loading.
    """
    try:
        # local import to avoid hard dep at import time
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Provide a concise, helpful answer."},
                {"role": "user", "content": prompt},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (ImportError, OSError, ValueError, AttributeError):
        pass
    # Fallback to a generic instruction-style prompt
    return f"You are a helpful assistant.\n\nUser: {prompt}\nAssistant:"


def demo(model_name: str, prompt: str = "Hello world", max_tokens: int = 128):
    """
    Generate a short text completion using a specified Hugging Face model.

    Behavior
    --------
    - If vLLM is available (e.g., on Linux with GPU), use vLLM for fast inference.
    - Otherwise, fall back to Hugging Face Transformers on CPU (slower but portable).

    Parameters
    ----------
    model_name : str
        The Hugging Face model ID to load (e.g., "microsoft/Phi-3-mini-4k-instruct").
    prompt : str, optional
        The input prompt to condition generation on. Defaults to "Hello world".
    max_tokens : int, optional
        Maximum number of new tokens to generate. Defaults to 128.

    Side Effects
    ------------
    Prints the prompt and generated completion to standard output.

    Notes
    -----
    - Uses chat templates if available to get more coherent, instruction-following outputs.
    - Sampling tuned with temperature/top_p for richer text.
    - Transformers fallback runs on CPU; prefer a small model when not using GPU.
    """
    rendered = _render_prompt(model_name, prompt)

    if USE_VLLM:
        # Import here to avoid possibly-unbound names when vLLM is unavailable
        from vllm import LLM, SamplingParams  # type: ignore
        llm = LLM(model=model_name)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        outputs = llm.generate([rendered], sampling_params)

        text = outputs[0].outputs[0].text
    else:
        # Transformers CPU fallback
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        generate = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # CPU
        )
        result = generate(
            rendered,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
        text = result["generated_text"]

    print("Prompt:", prompt)
    print("Completion:", text.strip())


if __name__ == "__main__":
    # Replace with one of your chosen small models:
    demo("microsoft/Phi-3-mini-4k-instruct", "What is your name?", 128)
    # or try a smaller CPU-friendly option:
    # demo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
