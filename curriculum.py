"""
AURAX Evo — Curriculum Learning (curriculum.py)
================================================
Manages adaptive difficulty progression across training generations.

Strategy:
  1. Start with easy tasks (difficulty 0.0–0.35)
  2. Track model performance per difficulty tier
  3. Gradually shift distribution toward harder tasks as performance improves
  4. Never drop easy tasks entirely (consolidation)
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("Curriculum")


# ---------------------------------------------------------------------------
# Difficulty tiers
# ---------------------------------------------------------------------------
TIERS = ["easy", "medium", "hard"]

TIER_RANGES = {
    "easy":   (0.00, 0.35),
    "medium": (0.35, 0.65),
    "hard":   (0.65, 1.00),
}

# Minimum proportion per tier (never go below these)
MIN_PROPORTIONS = {"easy": 0.10, "medium": 0.15, "hard": 0.00}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TierStats:
    """Performance statistics for a single difficulty tier."""
    tier: str
    attempts: int = 0
    successes: int = 0

    @property
    def accuracy(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def record(self, success: bool):
        self.attempts += 1
        if success:
            self.successes += 1


@dataclass
class CurriculumState:
    """Serialisable state persisted across generations."""
    generation: int = 0
    proportions: dict = field(default_factory=lambda: {
        "easy": 0.60, "medium": 0.30, "hard": 0.10
    })
    tier_stats: dict = field(default_factory=lambda: {
        t: asdict(TierStats(tier=t)) for t in TIERS
    })
    history: list = field(default_factory=list)   # list of proportion dicts


# ---------------------------------------------------------------------------
# Curriculum manager
# ---------------------------------------------------------------------------
class CurriculumManager:
    """
    Adaptive curriculum that shifts difficulty distribution based on
    model performance at each tier.

    Usage:
        cm = CurriculumManager(state_path="curriculum_state.json")
        weights = cm.get_weights()          # {easy: 0.6, medium: 0.3, hard: 0.1}
        cm.record_result("easy", success=True)
        cm.step_generation()                # call after each generation
        cm.save()
    """

    # Thresholds to "graduate" from a tier
    GRADUATION_ACCURACY = 0.75   # model must hit 75 % on a tier to unlock harder tasks
    SHIFT_STEP = 0.05            # proportion shift per graduation event

    def __init__(
        self,
        state_path: str = "curriculum_state.json",
        progression: str = "adaptive",   # "adaptive" | "linear"
        start_difficulty: float = 0.20,
    ):
        self.state_path = Path(state_path)
        self.progression = progression
        self.start_difficulty = start_difficulty
        self.state = self._load_or_init()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load_or_init(self) -> CurriculumState:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                state = CurriculumState(**data)
                logger.info(
                    f"Curriculum loaded (gen={state.generation}, "
                    f"props={state.proportions})"
                )
                return state
            except Exception as exc:
                logger.warning(f"Could not load curriculum state: {exc}. Resetting.")

        # Fresh state — bias toward easy
        init_props = self._init_proportions()
        state = CurriculumState(proportions=init_props)
        logger.info(f"Curriculum initialised: {init_props}")
        return state

    def _init_proportions(self) -> dict:
        """Derive initial proportions from start_difficulty."""
        easy_frac = max(0.4, 1.0 - self.start_difficulty)
        hard_frac = min(0.1, self.start_difficulty * 0.3)
        medium_frac = 1.0 - easy_frac - hard_frac
        return {
            "easy":   round(easy_frac,   2),
            "medium": round(medium_frac, 2),
            "hard":   round(hard_frac,   2),
        }

    def save(self):
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2))
        logger.debug(f"Curriculum state saved to {self.state_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_weights(self) -> dict:
        """Return current sampling weights {tier: proportion}."""
        return dict(self.state.proportions)

    def record_result(self, tier: str, success: bool):
        """Record a model evaluation result for a given difficulty tier."""
        if tier not in self.state.tier_stats:
            self.state.tier_stats[tier] = asdict(TierStats(tier=tier))
        stats = self.state.tier_stats[tier]
        stats["attempts"] += 1
        if success:
            stats["successes"] += 1

    def record_batch_results(self, results: list[dict]):
        """
        Convenience: record many results at once.
        Each result: {"tier": "easy", "success": True}
        """
        for r in results:
            self.record_result(r["tier"], r["success"])

    def step_generation(self):
        """
        Call after each generation. Updates proportions based on
        current tier accuracy and saves state.
        """
        self.state.generation += 1

        if self.progression == "adaptive":
            self._adaptive_update()
        elif self.progression == "linear":
            self._linear_update()

        # Snapshot history
        self.state.history.append({
            "generation": self.state.generation,
            "proportions": dict(self.state.proportions),
        })
        self.save()
        logger.info(
            f"[Gen {self.state.generation}] Curriculum updated: "
            f"{self.state.proportions}"
        )

    # ------------------------------------------------------------------
    # Update strategies
    # ------------------------------------------------------------------
    def _adaptive_update(self):
        """
        Shift distribution toward harder tiers when the model
        demonstrates mastery at easier tiers.
        """
        props = self.state.proportions
        stats = self.state.tier_stats

        def acc(tier: str) -> float:
            s = stats.get(tier, {})
            if s.get("attempts", 0) == 0:
                return 0.0
            return s["successes"] / s["attempts"]

        easy_acc   = acc("easy")
        medium_acc = acc("medium")

        # Graduate easy → medium
        if easy_acc >= self.GRADUATION_ACCURACY and props["easy"] > MIN_PROPORTIONS["easy"]:
            shift = min(self.SHIFT_STEP, props["easy"] - MIN_PROPORTIONS["easy"])
            props["easy"]   -= shift
            props["medium"] += shift / 2
            props["hard"]   += shift / 2
            logger.info(f"  Graduated from easy (acc={easy_acc:.2f}): -{shift:.2f} easy")

        # Graduate medium → hard
        if medium_acc >= self.GRADUATION_ACCURACY and props["medium"] > MIN_PROPORTIONS["medium"]:
            shift = min(self.SHIFT_STEP, props["medium"] - MIN_PROPORTIONS["medium"])
            props["medium"] -= shift
            props["hard"]   += shift
            logger.info(f"  Graduated from medium (acc={medium_acc:.2f}): -{shift:.2f} medium")

        # Normalise to sum=1
        total = sum(props.values())
        self.state.proportions = {k: round(v / total, 4) for k, v in props.items()}

    def _linear_update(self):
        """
        Simple linear progression: reduce easy by SHIFT_STEP each generation,
        increase hard proportionally.
        """
        props = self.state.proportions
        shift = self.SHIFT_STEP

        if props["easy"] - shift >= MIN_PROPORTIONS["easy"]:
            props["easy"]   -= shift
            props["hard"]   += shift * 0.6
            props["medium"] += shift * 0.4

        total = sum(props.values())
        self.state.proportions = {k: round(v / total, 4) for k, v in props.items()}

    # ------------------------------------------------------------------
    # Helpers / reporting
    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"=== Curriculum Summary (Generation {self.state.generation}) ===",
            f"Sampling weights: {self.state.proportions}",
            "",
            "Tier performance:",
        ]
        for tier in TIERS:
            s = self.state.tier_stats.get(tier, {})
            att = s.get("attempts", 0)
            suc = s.get("successes", 0)
            acc = (suc / att * 100) if att > 0 else 0.0
            lines.append(f"  {tier:8s}: {suc}/{att} ({acc:.1f}%)")
        return "\n".join(lines)

    def reset_tier_stats(self):
        """Reset per-tier stats at the start of each generation."""
        self.state.tier_stats = {
            t: asdict(TierStats(tier=t)) for t in TIERS
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="AURAX Evo — Curriculum Manager")
    parser.add_argument("--state", default="curriculum_state.json")
    parser.add_argument("--progression", default="adaptive", choices=["adaptive", "linear"])
    parser.add_argument("--start-difficulty", type=float, default=0.20)
    parser.add_argument("--summary", action="store_true", help="Print current summary")
    parser.add_argument("--step", action="store_true", help="Simulate one generation step")
    args = parser.parse_args()

    cm = CurriculumManager(
        state_path=args.state,
        progression=args.progression,
        start_difficulty=args.start_difficulty,
    )

    if args.summary:
        print(cm.summary())

    if args.step:
        cm.step_generation()
        print(cm.summary())


if __name__ == "__main__":
    main()
