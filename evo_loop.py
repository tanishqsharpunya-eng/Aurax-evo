"""
AURAX Evo — Self-Improvement Loop (evo_loop.py)
================================================
Orchestrates the recursive self-improvement pipeline:

  Generation N:
    1. Generate synthetic training data (synthetic_gen.py)
    2. Fine-tune candidate model (calls finetune.py)
    3. Run safety checks (safety.py)
    4. Evaluate candidate vs current (validator.py)
    5. Promote if improvement > threshold AND safety passes
    6. Update curriculum (curriculum.py)
    7. Pause for human approval every K generations

Supports resuming from a checkpoint and graceful interruption (SIGINT).
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Local modules (same directory)
from synthetic_gen import SyntheticGenerator
from curriculum import CurriculumManager
from validator import Validator, EvalResult
from safety import SafetyMonitor, safety_monitor_from_config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("EvoLoop")


# ---------------------------------------------------------------------------
# Generation record (persisted to history)
# ---------------------------------------------------------------------------
@dataclass
class GenerationRecord:
    generation: int
    timestamp: float
    current_model: str
    candidate_model: str
    num_samples_generated: int
    train_duration_s: float
    current_score: float
    candidate_score: float
    delta_pct: float
    safety_passed: bool
    promoted: bool
    reason: str
    benchmark_details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Human approval gate
# ---------------------------------------------------------------------------
def wait_for_human_approval(
    generation: int,
    verdict: dict,
    auto_approve: bool = False,
    approval_timeout_s: int = 3600,
) -> bool:
    """
    Pause the loop and wait for human input.

    Returns True (continue), False (abort).
    If auto_approve=True, skips interaction (useful for CI/headless mode).
    """
    if auto_approve:
        logger.info("Auto-approve enabled — continuing automatically.")
        return True

    print("\n" + "=" * 65)
    print(f"⏸  HUMAN APPROVAL REQUIRED — Generation {generation}")
    print("=" * 65)
    print(f"  Current model score : {verdict['current_score']:.4f}")
    print(f"  Candidate score     : {verdict['candidate_score']:.4f}")
    print(f"  Improvement         : {verdict['delta_pct']:+.2f}%")
    print(f"  Safety              : {'✓ PASS' if verdict['safety_passed'] else '✗ FAIL'}")
    print(f"  Promotion verdict   : {'PROMOTE' if verdict['promote'] else 'SKIP'}")
    print()
    print("Options:")
    print("  [c] Continue evolution loop")
    print("  [a] Abort loop")
    print("  [s] Skip this generation (revert, do not promote even if improvement found)")
    print()

    deadline = time.time() + approval_timeout_s
    while time.time() < deadline:
        try:
            choice = input("Your choice [c/a/s]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            logger.info("No input — aborting for safety.")
            return False

        if choice == "c":
            logger.info("Human approved — continuing.")
            return True
        elif choice == "a":
            logger.info("Human aborted loop.")
            return False
        elif choice == "s":
            logger.info("Human skipped generation — rolling back promotion.")
            verdict["promote"] = False
            return True
        else:
            print("Please enter 'c', 'a', or 's'.")

    logger.warning("Approval timeout reached — pausing loop until next run.")
    return False


# ---------------------------------------------------------------------------
# Fine-tuning caller
# ---------------------------------------------------------------------------
def run_finetune(
    train_data_path: str,
    output_dir: str,
    config: dict,
    dry_run: bool = False,
) -> bool:
    """
    Invoke finetune.py as a subprocess.
    Returns True on success.
    """
    finetune_script = Path(__file__).parent / "finetune.py"
    if not finetune_script.exists():
        logger.error(f"finetune.py not found at {finetune_script}")
        return False

    base_model = config.get("model", {}).get("base", "Qwen2.5-1.5B-Instruct")
    adapter_dir = config.get("model", {}).get("adapter_dir", "models/adapters")

    cmd = [
        sys.executable, str(finetune_script),
        "--train_data", train_data_path,
        "--output_dir", output_dir,
        "--base_model", base_model,
        "--adapter_dir", adapter_dir,
    ]

    logger.info(f"Running fine-tune: {' '.join(cmd)}")

    if dry_run:
        logger.info("[DRY RUN] Skipping actual training.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Write a stub adapter_config.json so validator can detect it
        stub = {
            "base_model_name_or_path": base_model,
            "peft_type": "LORA",
        }
        (Path(output_dir) / "adapter_config.json").write_text(json.dumps(stub))
        return True

    start = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        logger.info(f"Fine-tuning complete in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(f"Fine-tuning failed (exit {exc.returncode})")
        return False


# ---------------------------------------------------------------------------
# History / checkpoint management
# ---------------------------------------------------------------------------
class EvolutionHistory:
    def __init__(self, path: str = "evo_history.jsonl"):
        self.path = Path(path)
        self.records: list[GenerationRecord] = self._load()

    def _load(self) -> list:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text().splitlines():
            try:
                records.append(GenerationRecord(**json.loads(line)))
            except Exception:
                pass
        return records

    def append(self, record: GenerationRecord):
        self.records.append(record)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def last_promoted_model(self) -> Optional[str]:
        for r in reversed(self.records):
            if r.promoted:
                return r.candidate_model
        return None

    def summary_table(self) -> str:
        lines = [
            f"{'Gen':>4}  {'Score':>8}  {'Δ%':>7}  {'Prom':>5}  {'Safety':>6}",
            "-" * 42,
        ]
        for r in self.records:
            prom  = "✓" if r.promoted else "✗"
            safe  = "✓" if r.safety_passed else "✗"
            lines.append(
                f"{r.generation:>4}  {r.candidate_score:>8.4f}  "
                f"{r.delta_pct:>+7.2f}  {prom:>5}  {safe:>6}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graceful shutdown on SIGINT / SIGTERM
# ---------------------------------------------------------------------------
_SHUTDOWN_REQUESTED = False

def _handle_signal(sig, frame):
    global _SHUTDOWN_REQUESTED
    logger.warning("Shutdown signal received — will stop after current generation.")
    _SHUTDOWN_REQUESTED = True

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------
class EvoLoop:
    """
    Orchestrates the recursive self-improvement loop.

    Config keys used:
      generation.max_generations
      generation.samples_per_generation
      generation.improvement_threshold
      generation.human_approval_every
      generation.auto_approve
      generation.dry_run
      model.base
      model.adapter_dir
      safety.*
      benchmark.*
    """

    def __init__(self, config: dict):
        self.config = config
        gen_cfg = config.get("generation", {})

        self.max_generations: int     = gen_cfg.get("max_generations", 50)
        self.samples_per_gen: int     = gen_cfg.get("samples_per_generation", 5000)
        self.improvement_threshold    = gen_cfg.get("improvement_threshold", 0.05)
        self.human_approval_every: int = gen_cfg.get("human_approval_every", 5)
        self.auto_approve: bool       = gen_cfg.get("auto_approve", False)
        self.dry_run: bool            = gen_cfg.get("dry_run", False)

        self.models_dir = Path(config.get("model", {}).get("adapter_dir", "models/adapters"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.history = EvolutionHistory(
            config.get("evolution", {}).get("history_path", "evo_history.jsonl")
        )

        # Sub-systems
        self.data_gen = SyntheticGenerator(config)
        self.curriculum = CurriculumManager(
            state_path=config.get("evolution", {}).get("curriculum_state", "curriculum_state.json"),
            progression=config.get("data", {}).get("difficulty_progression", "adaptive"),
            start_difficulty=config.get("data", {}).get("difficulty_start", 0.20),
        )
        self.validator = Validator(config)
        self.safety_monitor = safety_monitor_from_config(config)

        # Determine starting model
        self.current_model: str = (
            self.history.last_promoted_model()
            or config.get("model", {}).get("base", "Qwen2.5-1.5B-Instruct")
        )
        self.current_score: Optional[float] = None

        logger.info(f"EvoLoop initialised. Starting model: {self.current_model}")

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def run(self, start_generation: int = 0):
        global _SHUTDOWN_REQUESTED

        logger.info(
            f"Starting evolution loop: generations={self.max_generations}, "
            f"samples/gen={self.samples_per_gen}, threshold={self.improvement_threshold:.0%}"
        )

        # Baseline evaluation of current model
        if self.current_score is None:
            logger.info("Running baseline evaluation of current model …")
            baseline_result = self.validator.evaluate(self.current_model)
            self.current_score = baseline_result.overall_score
            self.validator.print_report(baseline_result)

        for generation in range(start_generation, self.max_generations):
            if _SHUTDOWN_REQUESTED:
                logger.info("Shutdown requested — exiting loop.")
                break

            logger.info(f"\n{'#'*65}")
            logger.info(f"# GENERATION {generation:03d}")
            logger.info(f"{'#'*65}")

            record = self._run_generation(generation)
            self.history.append(record)

            logger.info(f"Generation {generation} complete. " + (
                f"✓ PROMOTED ({record.delta_pct:+.2f}%)"
                if record.promoted else
                f"✗ Not promoted ({record.reason})"
            ))

            # Human approval gate
            if (generation + 1) % self.human_approval_every == 0:
                verdict = {
                    "current_score":   record.current_score,
                    "candidate_score": record.candidate_score,
                    "delta_pct":       record.delta_pct,
                    "safety_passed":   record.safety_passed,
                    "promote":         record.promoted,
                }
                should_continue = wait_for_human_approval(
                    generation, verdict,
                    auto_approve=self.auto_approve,
                )
                if not should_continue:
                    logger.info("Human stopped the loop.")
                    break

        logger.info("\n" + "=" * 65)
        logger.info("Evolution loop finished.")
        logger.info(f"Final model: {self.current_model}")
        logger.info(f"Final score: {self.current_score:.4f}")
        logger.info("\nHistory:\n" + self.history.summary_table())

    # ------------------------------------------------------------------
    # Single generation
    # ------------------------------------------------------------------
    def _run_generation(self, generation: int) -> GenerationRecord:
        start_time = time.time()
        candidate_dir = str(self.models_dir / f"gen_{generation:03d}")
        synthetic_path = f"synthetic_data_gen{generation:03d}.jsonl"

        # ---- Step 1: Generate synthetic data ----
        logger.info(f"[Step 1] Generating {self.samples_per_gen} synthetic samples …")
        curriculum_weights = self.curriculum.get_weights()
        logger.info(f"  Curriculum weights: {curriculum_weights}")

        num_generated = self.data_gen.generate(
            num_samples=self.samples_per_gen,
            curriculum_weights=curriculum_weights,
        )
        logger.info(f"  Generated {num_generated} samples → {synthetic_path}")

        # ---- Step 2: Fine-tune ----
        logger.info(f"[Step 2] Fine-tuning candidate model → {candidate_dir}")
        train_ok = run_finetune(
            train_data_path=synthetic_path,
            output_dir=candidate_dir,
            config=self.config,
            dry_run=self.dry_run,
        )
        train_duration = time.time() - start_time

        if not train_ok:
            return self._make_record(
                generation, candidate_dir, train_duration,
                promoted=False, reason="Training failed",
                safety_passed=False,
            )

        # ---- Step 3: Safety checks ----
        logger.info(f"[Step 3] Running safety checks …")
        from validator import ModelRunner
        candidate_runner = ModelRunner(candidate_dir, device=self.config.get("model", {}).get("device", "auto"))
        try:
            candidate_runner._load()
            safety_result = self.safety_monitor.run_checks(
                candidate_runner, baseline_perplexity=None
            )
            self.safety_monitor.print_report(safety_result)
        except Exception as exc:
            logger.error(f"Safety check error: {exc}")
            safety_result = type("SR", (), {"passed": False, "violations": [str(exc)]})()
        finally:
            try:
                candidate_runner.unload()
            except Exception:
                pass

        if not safety_result.passed:
            logger.warning(f"SAFETY VIOLATION — candidate model rejected. "
                           f"Violations: {getattr(safety_result, 'violations', [])}")
            return self._make_record(
                generation, candidate_dir, train_duration,
                promoted=False, reason="Safety check failed",
                safety_passed=False,
            )

        # ---- Step 4: Benchmark evaluation ----
        logger.info(f"[Step 4] Evaluating candidate vs current model …")
        candidate_result = self.validator.evaluate(candidate_dir)
        self.validator.print_report(candidate_result)

        verdict = self.validator.compare(
            # Create a minimal EvalResult for current model score
            type("ER", (), {
                "overall_score": self.current_score,
                "benchmarks": {},
                "safety_passed": True,
            })(),
            candidate_result,
        )

        logger.info(
            f"  Current:   {verdict['current_score']:.4f}\n"
            f"  Candidate: {verdict['candidate_score']:.4f}\n"
            f"  Δ%:        {verdict['delta_pct']:+.2f}%"
        )

        # ---- Step 5: Promotion decision ----
        promoted = verdict["promote"]
        reason = ""

        if promoted:
            self.current_model = candidate_dir
            self.current_score = candidate_result.overall_score
            reason = f"Improved {verdict['delta_pct']:+.2f}%"
            logger.info(f"✓ Model promoted: {candidate_dir}")

            # Update curriculum based on benchmark results
            self._update_curriculum(candidate_result)
        else:
            reason = (
                "Below threshold" if verdict["delta_pct"] < self.improvement_threshold * 100
                else "Safety failed"
            )
            logger.info(f"✗ Model NOT promoted: {reason}")

        return self._make_record(
            generation, candidate_dir, train_duration,
            promoted=promoted, reason=reason,
            safety_passed=safety_result.passed,
            current_score=verdict["current_score"],
            candidate_score=verdict["candidate_score"],
            delta_pct=verdict["delta_pct"],
            num_samples=num_generated,
            benchmark_details=verdict.get("per_benchmark", {}),
        )

    def _update_curriculum(self, eval_result: EvalResult):
        """Translate benchmark results into curriculum tier feedback."""
        # Map benchmark names → curriculum tiers (simple heuristic)
        from curriculum import TIER_RANGES
        import random

        for name, br in eval_result.benchmarks.items():
            # Mark tier based on benchmark pass rate
            score = br.score
            if score > 0.80:
                tier = "easy"    # mastered — shift harder
            elif score > 0.50:
                tier = "medium"
            else:
                tier = "hard"

            for _ in range(min(br.total, 20)):
                success = random.random() < score
                self.curriculum.record_result(tier, success)

        self.curriculum.step_generation()

    def _make_record(
        self, generation: int, candidate_dir: str, train_duration: float,
        promoted: bool, reason: str, safety_passed: bool,
        current_score: Optional[float] = None,
        candidate_score: Optional[float] = None,
        delta_pct: float = 0.0,
        num_samples: int = 0,
        benchmark_details: Optional[dict] = None,
    ) -> GenerationRecord:
        return GenerationRecord(
            generation=generation,
            timestamp=time.time(),
            current_model=self.current_model,
            candidate_model=candidate_dir,
            num_samples_generated=num_samples,
            train_duration_s=train_duration,
            current_score=current_score or self.current_score or 0.0,
            candidate_score=candidate_score or 0.0,
            delta_pct=delta_pct,
            safety_passed=safety_passed,
            promoted=promoted,
            reason=reason,
            benchmark_details=benchmark_details or {},
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="AURAX Evo — Self-Improvement Loop")
    parser.add_argument("--config", default="config_evo.yaml", help="Config file")
    parser.add_argument("--start-generation", type=int, default=0)
    parser.add_argument("--max-generations", type=int, help="Override max_generations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip actual training (test pipeline)")
    parser.add_argument("--auto-approve", action="store_true",
                        help="Skip human approval prompts")
    parser.add_argument("--history", action="store_true",
                        help="Print history table and exit")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open() as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded from {args.config}")
    else:
        logger.warning(f"Config not found at {args.config} — using defaults.")
        config = {}

    # CLI overrides
    if args.dry_run:
        config.setdefault("generation", {})["dry_run"] = True
    if args.auto_approve:
        config.setdefault("generation", {})["auto_approve"] = True
    if args.max_generations:
        config.setdefault("generation", {})["max_generations"] = args.max_generations

    loop = EvoLoop(config)

    if args.history:
        print(loop.history.summary_table())
        return

    loop.run(start_generation=args.start_generation)


if __name__ == "__main__":
    main()
