import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger("brain_rag")


@dataclass(frozen=True)
class Usage:
    operation: str
    model: str
    latency_ms: float
    estimated_cost_usd: float
    input_tokens: int = 0
    output_tokens: int = 0


def timed(operation: str) -> Callable[..., Usage]:
    start = time.perf_counter()

    def finish(*, model: str, cost: float, input_tokens: int = 0, output_tokens: int = 0) -> Usage:
        latency_ms = (time.perf_counter() - start) * 1000
        usage = Usage(operation, model, latency_ms, cost, input_tokens, output_tokens)
        logger.info(
            "model_call operation=%s model=%s latency_ms=%.2f estimated_cost_usd=%.8f "
            "input_tokens=%d output_tokens=%d",
            operation,
            model,
            usage.latency_ms,
            usage.estimated_cost_usd,
            input_tokens,
            output_tokens,
        )
        return usage

    return finish
