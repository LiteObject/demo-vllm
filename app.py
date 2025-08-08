"""
Minimal vLLM demo script.

This module provides a single helper, `demo`, which loads a Hugging Face
text generation model via vLLM and prints a short completion for a given
prompt. It uses vLLM's default model implementation and sampling parameters
for broad compatibility with general HF models.
"""

# Try to use vLLM if available; otherwise fall back to Transformers on CPU.
try:
    from vllm import LLM, SamplingParams  # type: ignore
    USE_VLLM = True
except Exception:  # vLLM not available (e.g., on Windows)
    USE_VLLM = False


def demo(model_name: str, prompt: str = "Hello world", max_tokens: int = 50):
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
        Maximum number of new tokens to generate. Defaults to 50.

    Side Effects
    ------------
    Prints the prompt and generated completion to standard output.

    Notes
    -----
    - vLLM path: uses default model implementation and `SamplingParams`.
    - Transformers fallback: runs on CPU; prefer a small model for testing on Windows (e.g., TinyLlama).
    - First run may download model weights from the Hugging Face Hub.

    Raises
    ------
    Exception
        Propagates any exceptions from underlying libraries if model loading or generation fails.
    """
    if USE_VLLM:
        llm = LLM(model=model_name)

        sampling_params = SamplingParams(max_tokens=max_tokens)
        outputs = llm.generate([prompt], sampling_params)

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
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
        text = result["generated_text"]

    print("Prompt:", prompt)
    print("Completion:", text)


if __name__ == "__main__":
    # Replace with one of your chosen small models:
    demo("microsoft/Phi-3-mini-4k-instruct")
    # or try a smaller CPU-friendly option:
    # demo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
