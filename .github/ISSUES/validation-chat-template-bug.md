# [Bug] Validation script produces empty responses - missing chat template

## Summary

The `validate_model.py` script produces empty or malformed responses when testing models that require chat templates (like DeepSeek-R1-Distill). This causes:

1. False positive test results (empty response passes as "not refused")
2. Inability to properly validate model behavior

## Steps to Reproduce

1. Train or load a model based on `huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2`
2. Run validation:
   ```bash
   python scripts/validate_model.py \
     --base-model huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2 \
     --model models/distrust-r1-distill-14b/checkpoint-10000
   ```
3. Observe empty responses for all prompts

## Expected Behavior

- Model should receive properly formatted prompts with chat template
- Responses should contain substantive content
- Tests should accurately reflect model behavior

## Actual Behavior

- `generate_fn` passes raw prompts without chat template formatting
- Model returns empty strings or truncated responses
- Censorship tests show 4/4 passed (100%) when responses are actually empty
- After fixing false positives, tests show 0/4 passed with empty responses

## Root Cause

In `scripts/validate_model.py`, the `generate_fn` wrapper (line ~277) does not apply the tokenizer's chat template:

```python
def generate_fn(model, tokenizer, prompt, max_tokens=200):
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
```

Models like DeepSeek-R1-Distill require prompts formatted as:

```
<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜>
```

Without this formatting, the model produces empty or garbage output.

## Proposed Fix

Update `generate_fn` to apply chat template when available:

```python
def generate_fn(model, tokenizer, prompt, max_tokens=200):
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{'role': 'user', 'content': prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted = prompt
    return generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens)
```

Also update censorship tests to require:

- Minimum response length
- Required topic keywords (at least 2)

## Tasks

- [x] Identify root cause
- [ ] Create branch: `fix/validation-chat-template`
- [ ] Implement chat template fix in `generate_fn`
- [ ] Keep improved validation tests (min_length, required_keywords)
- [ ] Test fix with DeepSeek model
- [ ] Open PR to main

## Related Files

- `scripts/validate_model.py` - Main validation script
- `scripts/evaluate_checkpoint.py` - Also needs chat template fix

## Environment

- mlx-lm version: 0.28.3
- Model: huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2
- Platform: macOS (Apple Silicon)
