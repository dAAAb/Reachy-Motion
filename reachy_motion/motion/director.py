"""The director: turns the robot's streaming speech text into gestures that land with the audio.

Voice backends call, from any thread:

* ``speech_text(delta, pos)`` — text the robot will say; ``pos`` is the speaker-stream position (s) where it will be
  heard (default: the end of what is queued now). Text is cut into clauses at punctuation.
* ``speech_line(line, pos)`` — a whole sentence at once (e.g. the Taiwanese TTS path, sentence by sentence).
* ``user_text(text)`` — what the person said (context for the planner) / ``listening()`` while they talk.
* ``end_of_turn()`` and ``interrupt()`` (barge-in).

For each clause a **reflex** gesture can fire immediately (cue words), then the **planner** writes a full recipe;
both are scheduled on the speaker's playback clock and dropped if they would arrive after the clause was spoken.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from reachy_motion.motion.planner import Gesture, GesturePlanner, reflex
from reachy_motion.motion.recipe import FPS, recipe_to_poses

logger = logging.getLogger(__name__)

# "." ends a sentence only when followed by whitespace (so "3.5" and a streamed "p." + "m." are not cut) and not
# after common abbreviations; a trailing "." at the very end of a turn is handled by end_of_turn().
HARD_STOP = re.compile(r"[。！？!?\n]+|(?<!\b[DM]r)(?<!\bMrs)(?<!\bMs)(?<!\bSt)(?<!\bvs)(?<!\b[ap]\.m)\.(?=\s)")
WORDY = re.compile(r"[\u3400-\u9fffA-Za-z0-9]")
TAG = re.compile(r"\[[^\]\n]{1,16}\]")  # Eleven v3/v4 audio tags like [溫柔] / [laughs] are not spoken
SOFT_STOP = re.compile(r"[，,；;、：:…—]")
CJK = re.compile(r"[㐀-鿿]")

LISTENING = "go .4 e=-5 p=-5 r=8 z=5 E=.6 | hold 3 E=.4 | go 1 r=4 E=.3 | hold 3 E=.3"


def spoken_seconds(text: str) -> float:
    """Rough speaking time: ~4.5 CJK chars/s, ~2.6 English words/s."""
    cjk = len(CJK.findall(text))
    words = len(re.findall(r"[A-Za-z0-9']+", text))
    return cjk / 4.5 + words / 2.6


@dataclass
class _Clause:
    text: str
    pos: float | Future  # stream position where the clause starts being heard (or a Future of it)


class Director:
    def __init__(
        self,
        play: Callable[[object], None],
        planner: GesturePlanner | None,
        time_of: Callable[[float], float],
        position: Callable[[], float],
        on_event: Callable[[dict], None] | None = None,
        is_busy: Callable[[], bool] | None = None,
        lead_s: float = 0.1,
        use_reflex: bool = True,
        late_grace_s: float = 0.4,
        min_clause_chars: int = 6,
        max_clause_chars: int = 36,
    ) -> None:
        """``play(motion)`` starts a motion now (``Animator.play``); ``time_of``/``position`` are the speaker clock."""
        self._play = play
        self.planner = planner
        self.time_of, self.position = time_of, position
        self.on_event = on_event or (lambda e: None)
        self.is_busy = is_busy or (lambda: False)
        self.lead_s = lead_s
        self.use_reflex = use_reflex
        self.late_grace_s = late_grace_s  # how long after its clause a planned gesture may still start
        self.min_chars, self.max_chars = min_clause_chars, max_clause_chars
        self._buf = ""
        self._buf_pos: float | None = None
        self._heard: str | None = None
        self._tone_segments: list[tuple[str, str]] = []  # (tag, text after it) from the full reply text
        self._tone: str | None = None  # latest tone tag seen in streamed text
        self._turn_open = False  # the robot is mid-reply
        self._reaction: Gesture | None = None  # anticipatory gesture planned from what the person is saying
        self._react_for = ""
        self._react_pending = False
        self._turn = 0  # bumped by interrupt(): gestures of an older turn are dropped
        self._lock = threading.RLock()  # re-entrant: speech_text -> _try_reflex -> _schedule
        self._heap: list[tuple[float, int, int, Gesture, float]] = []
        self._seq = itertools.count()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="planner")
        self._wake = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._scheduler, name="director", daemon=True)
        self._thread.start()

    # -- inputs ----------------------------------------------------------------------------------------------------
    def user_text(self, text: str) -> None:
        text = text.strip()
        if text:
            self._heard = text
            self.on_event({"type": "user", "text": text})
            self.anticipate(text)

    def anticipate(self, heard_so_far: str) -> None:
        """Plan the robot's *reaction* while the person is still talking (for voices whose text arrives late).

        Re-planned every ~6 new characters; the latest result is played the moment the reply starts.
        """
        heard_so_far = heard_so_far.strip()
        if self.planner is None or self._react_pending or len(heard_so_far) < 4:
            return
        if len(heard_so_far) - len(self._react_for) < 6 and heard_so_far != self._heard:
            return
        self._react_pending, self._react_for = True, heard_so_far
        turn = self._turn

        def work() -> None:
            try:
                g = self.planner.react(heard_so_far)
            finally:
                self._react_pending = False
            if g is not None and turn == self._turn:
                self._reaction = g

        self._pool.submit(work)

    def set_reply_text(self, text: str) -> None:
        """The full reply text with its voice tags (e.g. ElevenLabs ``agent_response``): tags become planner hints."""
        parts = re.split(r"\[([^\]\n]{1,16})\]", text)
        self._tone_segments = [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
        if self._tone_segments:
            self._tone = self._tone_segments[0][0]

    def _tone_for(self, clause: str) -> str | None:
        head = WORDY.findall(clause)[:4]
        key = "".join(head)
        for tag, seg in self._tone_segments:
            if key and key in "".join(WORDY.findall(seg)):
                return tag
        return self._tone

    def listening(self) -> None:
        """The person started talking: perk up and lean in (only if nothing else is playing)."""
        self._play_recipe(Gesture(LISTENING, "listening", "reflex"), force=False)

    def speech_text(self, delta: str, pos: float | None = None) -> None:
        if not delta:
            return
        with self._lock:
            tags = TAG.findall(delta)
            if tags:
                self._tone = tags[-1].strip("[]")
            delta = TAG.sub("", delta)
            if self._buf_pos is None and delta.strip():
                self._buf_pos = self.position() if pos is None else pos
                first_of_turn = not self._turn_open
                self._turn_open = True
                fired = self.use_reflex and self._try_reflex(delta, self._buf_pos)
                if first_of_turn and not fired and self._reaction is not None:
                    self._schedule(self._reaction, self._buf_pos, self._buf_pos + 2.0)
                if first_of_turn:
                    self._reaction, self._react_for = None, ""
            self._buf += delta
            self._cut_clauses()

    def speech_line(self, line: str, pos: float | Future | None = None) -> None:
        """A complete sentence whose audio starts at ``pos``.

        ``pos`` may be a ``Future`` that resolves once the audio is synthesised and queued, so planning can run in
        parallel with TTS and the gesture is still placed exactly on the sentence's audio.
        """
        if isinstance(pos, Future):
            turn = self._turn

            def bind() -> None:
                try:
                    p = pos.result(timeout=30)
                except Exception:  # noqa: BLE001 - TTS failed: nothing will be said, nothing to gesture
                    return
                if turn == self._turn and self.use_reflex:
                    self._try_reflex(line, p, turn)

            self._pool.submit(bind)
            self._submit(_Clause(line, pos))
            return
        line = TAG.sub("", line).strip()
        if not WORDY.search(line):
            return
        p = self.position() if pos is None else pos
        if self.use_reflex:
            self._try_reflex(line, p)
        self._submit(_Clause(line, p))

    def end_of_turn(self) -> None:
        with self._lock:
            self._flush_buf()
            self._turn_open = False
            self._tone_segments, self._tone = [], None

    def interrupt(self) -> None:
        with self._lock:
            self._turn += 1
            self._buf, self._buf_pos = "", None
            self._turn_open = False
            self._heap.clear()
        self.on_event({"type": "interrupt"})

    def close(self) -> None:
        self._running = False
        self._wake.set()
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- clause cutting --------------------------------------------------------------------------------------------
    def _cut_clauses(self) -> None:
        while True:
            hard = HARD_STOP.search(self._buf)
            cut = None
            if hard:
                cut = hard.end()
            else:
                for m in SOFT_STOP.finditer(self._buf):
                    if len(self._buf[: m.end()].strip()) >= self.min_chars:
                        cut = m.end()
                        break
                if cut is None and len(self._buf) >= self.max_chars:
                    cut = len(self._buf)
            if cut is None:
                return
            text, rest = self._buf[:cut], self._buf[cut:]
            if not WORDY.search(text):  # punctuation only: glue it to the previous clause, nothing to plan
                self._buf = rest
                continue
            start = self._buf_pos if self._buf_pos is not None else self.position()
            # the rest starts roughly where this clause's spoken time ends
            self._buf = rest
            self._buf_pos = (start + spoken_seconds(text)) if rest.strip() else None
            self._submit(_Clause(text.strip(), start))

    def _flush_buf(self) -> None:
        if WORDY.search(self._buf):
            self._submit(_Clause(self._buf.strip(), self._buf_pos if self._buf_pos is not None else self.position()))
        self._buf, self._buf_pos = "", None

    # -- planning & scheduling -------------------------------------------------------------------------------------
    def _try_reflex(self, text: str, pos: float, turn: int | None = None) -> bool:
        g = reflex(text)
        if g is not None:
            self._schedule(g, pos, deadline_pos=pos + 1.0, turn=turn)
        return g is not None

    def _submit(self, clause: _Clause) -> None:
        turn = self._turn
        heard = self._heard
        tone = self._tone_for(clause.text)
        self.on_event({"type": "clause", "text": clause.text, "tone": tone})
        if self.planner is None:
            return

        def work() -> None:
            g = self.planner.plan(clause.text, heard, tone=tone)
            if turn != self._turn:
                return
            pos = clause.pos
            if isinstance(pos, Future):
                try:
                    pos = pos.result(timeout=30)
                except Exception:  # noqa: BLE001
                    return
            # still worth playing while the clause is being said (and a beat after it)
            deadline_pos = pos + max(1.0, spoken_seconds(clause.text) + self.late_grace_s)
            self._schedule(g, pos, deadline_pos, turn=turn)

        self._pool.submit(work)

    def _schedule(self, g: Gesture, pos: float, deadline_pos: float, turn: int | None = None) -> None:
        """Queue ``g``; ``turn`` = the turn it was planned for (a result racing an interrupt() is then dropped)."""
        at = self.time_of(pos) - self.lead_s
        deadline = self.time_of(deadline_pos)
        with self._lock:
            heapq.heappush(self._heap, (at, next(self._seq), self._turn if turn is None else turn, g, deadline))
        self._wake.set()

    def _scheduler(self) -> None:
        while self._running:
            now = time.monotonic()
            due: list[tuple[Gesture, float]] = []
            with self._lock:
                while self._heap and self._heap[0][0] <= now:
                    _, _, turn, g, deadline = heapq.heappop(self._heap)
                    if turn == self._turn:
                        due.append((g, deadline))
                wait = (self._heap[0][0] - now) if self._heap else 0.25
            for g, deadline in due:
                if now > deadline:
                    logger.debug("dropped late %s gesture for %r", g.source, g.line)
                    self.on_event({"type": "gesture_dropped", "source": g.source, "line": g.line})
                    continue
                self._play_recipe(g)
            self._wake.wait(max(0.005, min(wait, 0.25)))
            self._wake.clear()

    def _play_recipe(self, g: Gesture, force: bool = True) -> None:
        from reachy_animation import Clip

        try:
            poses = recipe_to_poses(g.recipe)
        except ValueError as e:
            logger.warning("bad recipe from %s: %s (%s)", g.source, g.recipe, e)
            return
        clip = Clip.from_frames(poses, FPS, name=f"{g.source}:{g.idea[:40]}")
        if not force and self.is_busy():
            return
        self._play(clip)
        self.on_event(
            {"type": "gesture", "source": g.source, "idea": g.idea, "recipe": g.recipe, "line": g.line,
             "latency_ms": round(g.latency_ms)}
        )
