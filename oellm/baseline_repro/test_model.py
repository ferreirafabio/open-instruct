#!/usr/bin/env python3
"""Simple test script to verify the trained models work."""

import argparse
import sys
import torch
from pathlib import Path
from datetime import datetime

# Add OLMo-core to path
sys.path.insert(0, "/work/dlclarge2/ferreira-oellm/OLMo-core/src")

from transformers import AutoTokenizer
from olmo_core.config import DType
from olmo_core.generate import GenerationConfig, TransformerGenerationModule
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.utils import get_default_device


def load_tokenizer_config_from_checkpoint(checkpoint_dir: str) -> TokenizerConfig:
    """Load tokenizer config from checkpoint's config.json."""
    import json
    
    config_path = Path(checkpoint_dir) / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)
    return TokenizerConfig.from_dict(config_dict["dataset"]["tokenizer"])


def test_model(checkpoint_dir: str, prompts: list[str], max_new_tokens: int = 256, output_file: str | None = None):
    """Test a model checkpoint with given prompts."""
    output_lines = []
    
    def log(msg: str = ""):
        print(msg)
        output_lines.append(msg)
    
    log(f"\n{'='*60}")
    log(f"Testing checkpoint: {checkpoint_dir}")
    log(f"Timestamp: {datetime.now().isoformat()}")
    log(f"{'='*60}\n")
    
    device = get_default_device()
    log(f"Using device: {device}")
    
    # Load tokenizer config and tokenizer
    tokenizer_config = load_tokenizer_config_from_checkpoint(checkpoint_dir)
    log(f"Tokenizer: {tokenizer_config.identifier}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.identifier)
    
    # Generation config
    generation_config = GenerationConfig(
        pad_token_id=tokenizer_config.pad_token_id,
        eos_token_id=tokenizer_config.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
    )
    
    log("Loading model...")
    generation_module = TransformerGenerationModule.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        generation_config=generation_config,
        device=device,
        dtype=DType.bfloat16,
    )
    log("Model loaded successfully!\n")
    
    results = []
    
    # Test each prompt
    for i, prompt in enumerate(prompts, 1):
        log(f"--- Prompt {i} ---")
        log(f"Input: {prompt}\n")
        
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
        
        with torch.no_grad():
            generated_ids, _, _ = generation_module.generate_batch(
                input_ids, completions_only=True, log_timing=True
            )
        
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        log(f"Output: {response}\n")
        log("-" * 40 + "\n")
        
        results.append({"prompt": prompt, "response": response})
    
    # Save output to file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(output_lines))
        print(f"\nOutput saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test trained models")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path")
    parser.add_argument("--name", type=str, default=None, help="Model name for default output file")
    args = parser.parse_args()
    
    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Write a short poem about coding.",
        "Explain machine learning in simple terms.",
    ]
    
    # Determine output file
    output_file = args.output
    if output_file is None and args.name:
        script_dir = Path(__file__).parent
        output_file = script_dir / f"test_output_{args.name}.txt"
    
    test_model(args.checkpoint, prompts, args.max_new_tokens, output_file)


if __name__ == "__main__":
    main()

