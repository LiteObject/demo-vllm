# Understanding Transformer Architecture

A beginner-friendly guide to how Transformers work.

## What Problem Do Transformers Solve?

Before Transformers, language models used **Recurrent Neural Networks (RNNs)** that processed text one word at a time, like reading a sentence left-to-right. This was slow and struggled with long-distance relationships between words.

**Transformers** changed everything by processing **all words at once** and letting each word "pay attention" to all other words in the sentence.

## The Big Idea: Self-Attention

Imagine you're reading the sentence:

> "The cat sat on the mat because **it** was tired."

To understand what "it" refers to, you need to look back at "cat." **Self-attention** is the mechanism that lets the model do exactly this—it computes how much each word should "attend to" (focus on) every other word.

### How Self-Attention Works

For each word in your input:

1. **Create three vectors**: Query (Q), Key (K), and Value (V)
   - Think of Q as "what I'm looking for"
   - K as "what I offer"
   - V as "the actual information I have"

2. **Compute attention scores**
   - Compare the Query of one word with the Keys of all other words
   - This tells us "how relevant is each word to this word?"
   - Formula: `Attention(Q, K, V) = softmax(Q·K^T / √d) · V`

3. **Weighted combination**
   - Use the scores to create a weighted sum of all the Values
   - Words with high scores contribute more to the final representation

### Example

For the word "it" in our sentence:
- Its Query vector asks: "Who am I referring to?"
- It compares against Keys from all words: "The", "cat", "sat", "on", "the", "mat"...
- "cat" has the highest score (most relevant)
- The output representation of "it" now contains information from "cat"

## The Transformer Building Blocks

### 1. Multi-Head Attention

Instead of one attention mechanism, Transformers use **multiple attention "heads"** in parallel.

- Each head can learn different relationships
  - Head 1 might learn subject-verb relationships
  - Head 2 might learn pronoun-antecedent relationships
  - Head 3 might learn semantic similarity

- Outputs from all heads are concatenated and transformed

### 2. Positional Encoding

Since attention processes all words simultaneously, the model doesn't know word order. **Positional encodings** add position information:

- Each position gets a unique vector
- Added to the word embedding before processing
- Allows the model to distinguish "dog bites man" from "man bites dog"

### 3. Feed-Forward Networks

After attention, each word representation passes through a simple neural network:

```
Input → Linear layer → ReLU activation → Linear layer → Output
```

This transforms the attention output and adds non-linearity.

### 4. Layer Normalization & Residual Connections

- **Residual connections**: Add the input back to the output (`output = layer(input) + input`)
  - Helps gradients flow during training
  - Prevents information loss

- **Layer normalization**: Stabilizes training by normalizing values across features

## Two Main Architectures

### Encoder-Decoder (Original Transformer)

Used for tasks where you transform one sequence into another (translation, summarization):

```
Input: "Hello world" (English)
         ↓
    [ENCODER] - Processes input with self-attention
         ↓
    [DECODER] - Generates output with cross-attention to encoder
         ↓
Output: "Bonjour le monde" (French)
```

### Decoder-Only (GPT, Llama, Phi-3)

Used for text generation (what this demo uses):

```
Input: "The cat sat on the"
         ↓
    [DECODER] - Self-attention + generation
         ↓
Output: "mat"
```

Key difference: **Causal masking**—each word can only attend to previous words, not future ones. This prevents "cheating" during generation.

## How Text Generation Works (Step-by-Step)

Let's generate text with a decoder-only model:

**Step 1: Initial prompt**
```
Input: "The cat"
→ Self-attention over "The" and "cat"
→ Predict next token: "sat" (probability: 0.85)
```

**Step 2: Add predicted word**
```
Input: "The cat sat"
→ Self-attention over all three words
→ Predict next token: "on" (probability: 0.72)
```

**Step 3: Continue**
```
Input: "The cat sat on"
→ Predict: "the"

Input: "The cat sat on the"
→ Predict: "mat"
```

This continues until you hit a stop token or max length.

## The KV Cache Optimization

**Problem**: Each new token requires attention over ALL previous tokens. For token 100, you recompute attention with tokens 1-99, even though they haven't changed.

**Solution**: Cache the Key and Value vectors from previous tokens.

- **Without cache**: Generate token 100 → recompute K/V for tokens 1-99 → O(n²) work
- **With cache**: Generate token 100 → reuse cached K/V → O(n) work

This is what makes modern LLM inference fast. vLLM's **PagedAttention** takes this further by managing the cache memory more efficiently.

## Why Transformers Won

1. **Parallelization**: Process entire sequences at once → faster training
2. **Long-range dependencies**: Attention connects distant words directly
3. **Scalability**: Architecture scales well to billions of parameters
4. **Versatility**: Same architecture works for text, images, audio, code, etc.

## Transformer Models You Might Know

- **GPT (Generative Pre-trained Transformer)**: Decoder-only, text generation
- **BERT**: Encoder-only, text understanding/classification
- **T5**: Encoder-decoder, text-to-text tasks
- **Llama, Phi-3, Mistral**: Modern decoder-only models with optimizations
- **Vision Transformers (ViT)**: Apply Transformers to images

## In This Demo

When you run:
```python
demo("microsoft/Phi-3-mini-4k-instruct", "What is AI?", 128)
```

Behind the scenes:
1. Tokenize "What is AI?" into subword tokens
2. Add positional encodings
3. Run through ~32 Transformer decoder layers (each with multi-head attention + feed-forward)
4. Generate next token probabilities
5. Sample a token, add to sequence, repeat
6. Use KV cache to avoid recomputing attention for past tokens

**vLLM** optimizes steps 3-6 with PagedAttention, continuous batching, and CUDA kernels.

## Further Reading

- Original paper: "Attention Is All You Need" (Vaswani et al., 2017)
- The Illustrated Transformer (Jay Alammar): https://jalammar.github.io/illustrated-transformer/
- Hugging Face Course: https://huggingface.co/learn/nlp-course/
