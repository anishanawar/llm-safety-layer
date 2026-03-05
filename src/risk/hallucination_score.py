from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import math


class HallucinationScorer:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def generate_samples(self, prompt, n_samples=5, max_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            max_new_tokens=max_tokens,
            num_return_sequences=n_samples
        )

        texts = [
            self.tokenizer.decode(out, skip_special_tokens=True)
            for out in outputs
        ]

        return texts

    def compute_entropy(self, samples):
        counts = {}
        for s in samples:
            counts[s] = counts.get(s, 0) + 1

        total = len(samples)
        probs = [count / total for count in counts.values()]

        entropy = -sum(p * math.log(p + 1e-9) for p in probs)
        return entropy

    def score(self, prompt):
        samples = self.generate_samples(prompt)

        entropy = self.compute_entropy(samples)

        # normalize entropy (rough heuristic)
        normalized_entropy = entropy / math.log(len(samples) + 1e-9)

        return {
            "samples": samples,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy
        }