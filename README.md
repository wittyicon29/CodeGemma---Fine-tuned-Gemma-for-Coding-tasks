# Fine-tuning Gemma-2B-it Using LoRA for Code Generation

## Introduction

Gemma is a family of lightweight, state-of-the-art open models from Google, built using the same research and technology as Gemini models. These models are **decoder-only** LLMs (Large Language Models) optimized for various text generation tasks, such as:
- **Question answering**
- **Summarization**
- **Reasoning**
- **Code generation**

Given their **relatively small size**, they can be deployed on resource-constrained environments such as personal laptops, desktops, and cloud-based servers. This project focuses on **fine-tuning Gemma-2B-it** using **LoRA (Low-Rank Adaptation)** to improve its **code generation capabilities**.

---
## Why Fine-tune?

To assess Gemma-2B-it’s **pre-trained code generation abilities**, I requested it to generate a **recursive Fibonacci function in Python**.

### Model's Response Before Fine-Tuning:
```python
 def fibonacci(n):
    """
    Calculates the nth number in the Fibonacci sequence.
    
    Args:
      n: The index of the Fibonacci number to calculate.
    
    Returns:
      The nth Fibonacci number.
    """
    
    # Base case: If n is 0 or 1, return the corresponding Fibonacci number.
    if n == 0:
        return 0
    elif n == 1:
        return 1
    
    # Calculate the previous two Fibonacci numbers.
    previous = fibonacci(n-1)
    next = fibonacci(n-2)
    
    # Return the Fibonacci number.
    return previous + next
```

While the response is technically correct, I expected a **more structured and optimized version**, such as the one from **GeeksforGeeks**, which provides a **clear function header, comments, and better readability**.

To improve the model’s ability to generate structured, high-quality code, I decided to **fine-tune it using LoRA** with a dedicated **code dataset**.

---
## Fine-tuning Setup

I fine-tuned **Gemma-2B-it** on **Google Colab (T4 GPU, 15GB VRAM)**. However, based on observations, this task requires only **~10GB VRAM**.

To optimize training, I used **QLoRA (Quantized LoRA)** with **4-bit quantization** to reduce memory usage while retaining performance.

### Dataset

The dataset chosen for fine-tuning is **[TokenBender/code_instructions_122k_alpaca_style](https://huggingface.co/datasets/TokenBender/code_instructions_122k_alpaca_style)**, which contains structured code generation instructions, ensuring **instruction-following abilities** are improved.

#### Example Dataset Sample:
```
### Instruction:
Write a Python function that calculates the factorial of a number recursively.

### Response:
```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n-1)
```
```

---
## Model Parameters & LoRA Configuration

To optimize training, I configured **LoRA** with appropriate hyperparameters. The configuration was inspired by [Adithya S K's guide on LoRA tuning](https://twitter.com/adithya_s_k/status/1744065797268656579).

### LoRA Configuration:
- **LoRA Rank (`r`)**: 8
- **LoRA Alpha**: 32
- **Dropout**: 0.1
- **Target Modules**: QKV (Query-Key-Value)
- **Quantization**: 4-bit using `bitsandbytes`

#### Total Trainable Parameters:
```
# LoRA introduces additional trainable parameters, significantly reducing memory consumption.
Total Trainable Parameters: ~3.5 million
```

---
## Training Process

The model was trained for **100 steps** with the following configuration:

### Training Parameters:
- **Batch Size**: 4
- **Gradient Accumulation Steps**: 16
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing

#### Training Loss Progression:
![Loss Graph](https://github.com/104-wonohfor/Finetune_LLM_Gemma-2b-it/assets/104601534/a8be0990-92a0-4258-aa8a-2abfa25e0b3d)

---
## Results

After fine-tuning, I tested the model again by requesting it to generate a recursive Fibonacci function.

### Model’s Response After Fine-Tuning:
```python
def fibonacci(n):
    """
    Returns the nth Fibonacci number using recursion.
    """
    if n <= 0:
        return "Input must be a positive integer."
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
n = 10
print("Fibonacci number at position", n, "is", fibonacci(n))
```

### Key Improvements:
✅ **More structured function definition**
✅ **Improved error handling**
✅ **Better docstrings**
✅ **Clearer example usage**

This result is much closer to the **GeeksforGeeks** standard, demonstrating that fine-tuning improved the model’s **code generation quality**.
---
## Future Work

🔹 **Train for more steps (e.g., 500–1000 steps) to further refine performance**
🔹 **Expand dataset with real-world coding problems**
🔹 **Use more advanced fine-tuning techniques like RLHF for better instruction following**
🔹 **Optimize inference speed with techniques like FasterTransformer**

---
## References
- [Gemma Models (Google)](https://ai.google.dev/gemma)
- [Hugging Face LoRA Documentation](https://huggingface.co/docs/peft/main/en/index)
- [TokenBender Code Dataset](https://huggingface.co/datasets/TokenBender/code_instructions_122k_alpaca_style)
