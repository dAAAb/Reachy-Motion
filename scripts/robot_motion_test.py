"""Real-robot motion smoke test (no voice): wake, play every reflex gesture, measure the 60 Hz set_target stream.

    python scripts/robot_motion_test.py [--host reachy-mini.local]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reachy_mini import ReachyMini  # noqa: E402
from reachy_animation import Clip, to_target  # noqa: E402

from reachy_motion.config import Settings  # noqa: E402
from reachy_motion.engine import Engine  # noqa: E402
from reachy_motion.motion.planner import REFLEXES  # noqa: E402
from reachy_motion.motion.recipe import FPS, recipe_to_poses  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    ap.add_argument("--gap", type=float, default=3.0, help="seconds between gestures")
    a = ap.parse_args()
    kw = {"media_backend": "no_media"}
    if a.host:
        kw["host"] = a.host
    robot = ReachyMini(**kw)
    engine = Engine(Settings.from_env(audio="local"), robot=robot)

    durations: list[float] = []
    stamps: list[float] = []
    errors = [0]
    original = robot.set_target

    def timed_set_target(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            original(*args, **kwargs)
        except Exception:
            errors[0] += 1
            raise
        finally:
            durations.append((time.perf_counter() - t0) * 1000)
            stamps.append(time.monotonic())

    robot.set_target = timed_set_target  # type: ignore[method-assign]
    engine.start_animation()
    print("awake; playing gestures:", ", ".join(name for _, name, _ in REFLEXES))
    for _, name, recipe in REFLEXES:
        engine.animator.play(Clip.from_frames(recipe_to_poses(recipe, seed=1), FPS, name=name))
        print(f"  {name}")
        time.sleep(a.gap)
    time.sleep(1.0)
    engine.close()
    engine.rest_robot()
    robot.__exit__(None, None, None)

    span = stamps[-1] - stamps[0] if len(stamps) > 1 else 0
    gaps = [b - a_ for a_, b in zip(stamps, stamps[1:])]
    print(f"\nset_target calls: {len(durations)} over {span:.1f}s = {len(durations) / span if span else 0:.1f} Hz, "
          f"errors: {errors[0]}")
    print(f"call duration ms: median {statistics.median(durations):.2f}  p95 {sorted(durations)[int(.95 * len(durations))]:.2f}"
          f"  max {max(durations):.2f}")
    print(f"tick gap ms: p95 {sorted(gaps)[int(.95 * len(gaps))] * 1000:.1f}  max {max(gaps) * 1000:.1f}")


if __name__ == "__main__":
    main()
