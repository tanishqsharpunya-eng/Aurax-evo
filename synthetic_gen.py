"""
AURAX Evo — Synthetic Data Generator (synthetic_gen.py)
========================================================
Generates diverse synthetic training data (Q&A pairs, code problems,
reasoning chains, vulnerability fixes) using the current AURAX model.

Supports:
  - Multiple domains: code, vulnerability, reasoning, qa
  - Hash-based deduplication
  - JSONL output with metadata
  - Configurable temperature, sample count, difficulty
"""

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("SyntheticGen")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    instruction: str
    input: str
    output: str
    domain: str
    difficulty: float          # 0.0 → 1.0
    generation_model: str
    timestamp: float
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            raw = f"{self.instruction}{self.input}{self.output}"
            self.hash = hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Prompt templates per domain
# ---------------------------------------------------------------------------
CODE_PROMPTS = [
    "Write a Python function that {task}. Include type hints, a docstring, and 3 pytest test cases.",
    "Implement a Python class for {task}. Add proper error handling and unit tests.",
    "Create a Python solution for: {task}. Optimize for readability and performance.",
]

VULN_PROMPTS = [
    "Show a Python code snippet with a {vuln_type} vulnerability, then provide the secure fixed version with an explanation of the flaw.",
    "Demonstrate a {vuln_type} security issue in Python and fix it. Explain why the original code is dangerous.",
]

REASONING_PROMPTS = [
    "Generate a {difficulty_label} math word problem and solve it step-by-step.",
    "Create a {difficulty_label} logical reasoning puzzle and show the complete solution.",
    "Write a {difficulty_label} algorithm problem and explain the solution with time/space complexity.",
]

QA_PROMPTS = [
    "Generate an insightful question about {topic} and provide a thorough, accurate answer.",
    "Ask and answer a {difficulty_label} question about {topic}. Be detailed and precise.",
]

CODE_TASKS = [
    "reverses a linked list", "finds all prime numbers up to N using a sieve",
    "implements a binary search tree with insert and search", "merges two sorted arrays",
    "computes the Fibonacci sequence iteratively and recursively",
    "validates an email address using regex", "flattens a nested list",
    "implements a least-recently-used (LRU) cache", "parses a simple CSV file without libraries",
    "counts word frequencies in a text string", "implements quicksort",
    "finds the longest common subsequence of two strings",
    "converts Roman numerals to integers", "detects a cycle in a directed graph",
    "implements a simple stack and queue using a list",
]

VULN_TYPES = [
    "SQL injection", "command injection", "path traversal", "XSS (cross-site scripting)",
    "insecure deserialization", "hardcoded credentials", "integer overflow",
    "race condition", "SSRF (server-side request forgery)", "buffer overflow simulation",
]

TOPICS = [
    "machine learning regularization", "quantum computing fundamentals",
    "distributed systems consistency models", "operating system scheduling",
    "TCP/IP networking", "cryptographic hash functions", "Docker containerization",
    "REST API design principles", "database indexing strategies",
    "garbage collection algorithms", "compiler design", "Kubernetes orchestration",
    "OAuth 2.0 authorization flows", "microservices patterns",
]

DIFFICULTY_LABELS = {
    "easy": (0.0, 0.35),
    "medium": (0.35, 0.65),
    "hard": (0.65, 1.0),
}


# ---------------------------------------------------------------------------
# Model backend (supports both Ollama and Transformers)
# ---------------------------------------------------------------------------
class ModelBackend:
    """Unified interface for either Ollama or HuggingFace Transformers."""

    def __init__(self, model_name: str, backend: str = "ollama", device: str = "auto"):
        self.model_name = model_name
        self.backend = backend
        self._model = None
        self._tokenizer = None

        if backend == "transformers":
            self._load_transformers(device)
        elif backend == "ollama":
            self._check_ollama()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _load_transformers(self, device: str):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"Loading {self.model_name} via transformers …")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device,
            )
            logger.info("Model loaded.")
        except ImportError:
            raise RuntimeError("Install transformers: pip install transformers torch")

    def _check_ollama(self):
        try:
            import ollama
            models = [m["name"] for m in ollama.list().get("models", [])]
            if not any(self.model_name in m for m in models):
                logger.warning(
                    f"Model '{self.model_name}' not found in Ollama. "
                    "Pull it with: ollama pull <model>"
                )
        except ImportError:
            raise RuntimeError("Install ollama: pip install ollama")

    def generate(self, prompt: str, temperature: float = 0.8, max_tokens: int = 1024) -> str:
        if self.backend == "ollama":
            return self._generate_ollama(prompt, temperature, max_tokens)
        return self._generate_transformers(prompt, temperature, max_tokens)

    def _generate_ollama(self, prompt: str, temperature: float, max_tokens: int) -> str:
        import ollama
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["response"].strip()

    def _generate_transformers(self, prompt: str, temperature: float, max_tokens: int) -> str:
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _difficulty_label(difficulty: float) -> str:
    for label, (lo, hi) in DIFFICULTY_LABELS.items():
        if lo <= difficulty < hi:
            return label
    return "hard"


def build_code_prompt(difficulty: float) -> tuple[str, str]:
    task = random.choice(CODE_TASKS)
    template = random.choice(CODE_PROMPTS)
    instruction = template.format(task=task)
    return instruction, ""


def build_vuln_prompt(difficulty: float) -> tuple[str, str]:
    vuln = random.choice(VULN_TYPES)
    template = random.choice(VULN_PROMPTS)
    instruction = template.format(vuln_type=vuln)
    return instruction, ""


def build_reasoning_prompt(difficulty: float) -> tuple[str, str]:
    label = _difficulty_label(difficulty)
    template = random.choice(REASONING_PROMPTS)
    instruction = template.format(difficulty_label=label)
    return instruction, ""


def build_qa_prompt(difficulty: float) -> tuple[str, str]:
    label = _difficulty_label(difficulty)
    topic = random.choice(TOPICS)
    template = random.choice(QA_PROMPTS)
    instruction = template.format(difficulty_label=label, topic=topic)
    return instruction, ""


DOMAIN_BUILDERS = {
    "code": build_code_prompt,
    "vulnerability": build_vuln_prompt,
    "reasoning": build_reasoning_prompt,
    "qa": build_qa_prompt,
}


# ---------------------------------------------------------------------------
# Deduplication store
# ---------------------------------------------------------------------------
class HashStore:
    def __init__(self, path: Path):
        self.path = path
        self._hashes: set[str] = set()
        if path.exists():
            self._hashes = set(path.read_text().splitlines())

    def contains(self, h: str) -> bool:
        return h in self._hashes

    def add(self, h: str):
        self._hashes.add(h)
        with self.path.open("a") as f:
            f.write(h + "\n")


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
class SyntheticGenerator:
    def __init__(self, config: dict):
        self.config = config
        gen_cfg = config.get("generation", {})
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        self.num_samples: int = gen_cfg.get("num_samples", 500)
        self.temperature: float = gen_cfg.get("temperature", 0.8)
        self.max_tokens: int = gen_cfg.get("max_tokens", 1024)
        self.output_path = Path(gen_cfg.get("output_path", "synthetic_data.jsonl"))
        self.domains: list[str] = data_cfg.get("domains", ["code", "reasoning", "qa"])

        model_name = model_cfg.get("base", "qwen2.5:1.5b")
        backend = model_cfg.get("backend", "ollama")
        device = model_cfg.get("device", "auto")

        self.backend = ModelBackend(model_name, backend=backend, device=device)
        self.model_name = model_name

        hash_path = self.output_path.with_suffix(".hashes")
        self.hash_store = HashStore(hash_path)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _sample_domain(self) -> str:
        return random.choice(self.domains)

    def _sample_difficulty(self, curriculum_weights: Optional[dict] = None) -> float:
        if curriculum_weights:
            labels = list(curriculum_weights.keys())
            weights = [curriculum_weights[l] for l in labels]
            label = random.choices(labels, weights=weights)[0]
            lo, hi = DIFFICULTY_LABELS[label]
            return random.uniform(lo, hi)
        return random.random()

    def generate_one(self, domain: str, difficulty: float) -> Optional[Sample]:
        """Generate a single training sample for the given domain and difficulty."""
        builder = DOMAIN_BUILDERS.get(domain)
        if builder is None:
            logger.warning(f"Unknown domain: {domain}")
            return None

        instruction, input_text = builder(difficulty)
        full_prompt = (
            f"You are an expert AI assistant. {instruction}\n\n"
            "Provide a complete, accurate, and well-structured response."
        )

        try:
            output = self.backend.generate(
                full_prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )
        except Exception as exc:
            logger.error(f"Generation failed: {exc}")
            return None

        if len(output.split()) < 20:
            logger.debug("Output too short, skipping.")
            return None

        sample = Sample(
            instruction=instruction,
            input=input_text,
            output=output,
            domain=domain,
            difficulty=difficulty,
            generation_model=self.model_name,
            timestamp=time.time(),
        )

        if self.hash_store.contains(sample.hash):
            logger.debug("Duplicate detected, skipping.")
            return None

        self.hash_store.add(sample.hash)
        return sample

    def generate(
        self,
        num_samples: Optional[int] = None,
        curriculum_weights: Optional[dict] = None,
        append: bool = False,
    ) -> int:
        """
        Generate num_samples synthetic samples and write to JSONL.

        Returns:
            Number of samples actually written.
        """
        n = num_samples or self.num_samples
        mode = "a" if append else "w"
        written = 0
        attempts = 0
        max_attempts = n * 3  # allow retries for duplicates/failures

        logger.info(f"Generating {n} samples across domains: {self.domains}")

        with self.output_path.open(mode) as f:
            while written < n and attempts < max_attempts:
                attempts += 1
                domain = self._sample_domain()
                difficulty = self._sample_difficulty(curriculum_weights)

                sample = self.generate_one(domain, difficulty)
                if sample is None:
                    continue

                f.write(json.dumps(asdict(sample)) + "\n")
                f.flush()
                written += 1

                if written % 50 == 0:
                    logger.info(f"  Progress: {written}/{n} samples written …")

        logger.info(f"Done. {written} samples written to {self.output_path}")
        return written


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="AURAX Evo — Synthetic Data Generator")
    parser.add_argument("--config", default="config_evo.yaml", help="Config file path")
    parser.add_argument("--num-samples", type=int, help="Override num_samples from config")
    parser.add_argument("--domains", nargs="+", help="Override domains from config")
    parser.add_argument("--output", help="Override output JSONL path")
    parser.add_argument("--append", action="store_true", help="Append to existing file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open() as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found at {args.config}, using defaults.")
        config = {}

    # CLI overrides
    if args.num_samples:
        config.setdefault("generation", {})["num_samples"] = args.num_samples
    if args.domains:
        config.setdefault("data", {})["domains"] = args.domains
    if args.output:
        config.setdefault("generation", {})["output_path"] = args.output

    gen = SyntheticGenerator(config)
    gen.generate(append=args.append)


if __name__ == "__main__":
    main()
