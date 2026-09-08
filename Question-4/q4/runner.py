"""
runner.py - Drive a whole passage through the live checks

The Streamlit app owns its own loop because it has to repaint the editor between
tokens. Everything else that needs a full passage run, which is the demo script
and the experiment sweep, shares this one so the numbers in the report come from
the same code path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from q4.grammar import GrammarResult, TRIGGER_N
from q4.pipeline import Alert, LiveDocument, check_token, check_window


@dataclass
class PassageRun:
    """Everything one pass over a passage produced."""

    document: LiveDocument
    alerts: list[Alert] = field(default_factory=list)
    token_latencies: list[float] = field(default_factory=list)
    trigger_latencies: list[float] = field(default_factory=list)
    live_flagged: list[bool] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: {"SEGMENT": 0, "SPELL": 0, "GRAMMAR": 0})
    wall_ms: float = 0.0

    @property
    def average_token_ms(self) -> float:
        return sum(self.token_latencies) / len(self.token_latencies) if self.token_latencies else 0.0

    @property
    def average_trigger_ms(self) -> float:
        return (
            sum(self.trigger_latencies) / len(self.trigger_latencies)
            if self.trigger_latencies
            else 0.0
        )


def run_passage(
    tokens: Sequence[tuple[str, bool]],
    models: dict[str, Any],
    trigger_n: int = TRIGGER_N,
    ppl_threshold: float | None = None,
    delay: float = 0.0,
    on_alert: Callable[[str, str, float], None] | None = None,
) -> PassageRun:
    """Type a whole passage token by token and collect what the checks found.

    `tokens` is the (token, sentence_end) stream the passage sampler produces.
    `on_alert` is called with the alert kind, its message and its latency as each
    one fires, which is how the demo script prints alerts in the order a user
    would have seen them.
    """
    run = PassageRun(document=LiveDocument())
    words_seen = 0
    next_trigger = trigger_n
    started = time.perf_counter()

    def flag_current() -> None:
        index = len(run.document.sentences)
        while len(run.live_flagged) <= index:
            run.live_flagged.append(False)
        run.live_flagged[index] = True

    for token, sentence_end in tokens:
        outcome = check_token(token, models)
        run.token_latencies.append(outcome.latency_ms)

        for alert in outcome.alerts:
            run.alerts.append(alert)
            run.counts[alert.kind] += 1
            flag_current()
            if on_alert:
                on_alert(alert.kind, alert.detail, alert.latency_ms)

        words_seen += len(outcome.words)
        run.document.add(outcome, force_sentence_end=sentence_end)

        if words_seen >= next_trigger:
            next_trigger += trigger_n
            window = run.document.words()[-trigger_n:]
            if len(window) >= 2:
                result = check_window(window, models, ppl_threshold=ppl_threshold)
                run.trigger_latencies.append(result.latency_ms)
                if result.fired:
                    run.counts["GRAMMAR"] += 1
                    flag_current()
                    if on_alert:
                        on_alert("GRAMMAR", describe_grammar(result), result.latency_ms)

        if delay:
            time.sleep(delay)

    run.document.close()
    while len(run.live_flagged) < len(run.document.sentences):
        run.live_flagged.append(False)
    run.wall_ms = (time.perf_counter() - started) * 1000
    return run


def describe_grammar(result: GrammarResult) -> str:
    """One line describing why a grammar trigger fired."""
    detail = f"window perplexity {result.ppl:.0f}"
    if result.real_word_fixes:
        fixes = ", ".join(f"{fix.original} -> {fix.suggestion}" for fix in result.real_word_fixes)
        detail += f", real-word suggestion {fixes}"
    return detail
