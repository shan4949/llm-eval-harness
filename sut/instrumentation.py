import statistics
import structlog
from dataclasses import dataclass
from dotenv import load_dotenv

from sut.generator import Generator, GeneratorResponse
from sut.retriever import RetrievedChunk
from evalharness.config import settings

load_dotenv()
log = structlog.get_logger()


@dataclass
class InstrumentedResponse:
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    estimated_cost_usd: float
    retrieved_context: str
    retrieval_sim_p50: float


class InstrumentedGenerator:
    def __init__(self):
        self.generator = Generator(model=settings.sut_model)

    def run(self, question: str, chunks: list[RetrievedChunk]) -> InstrumentedResponse:
        resp: GeneratorResponse = self.generator.generate(question, chunks)

        # cost from token usage
        pricing = settings.pricing.get(resp.model, {"input": 0.0, "output": 0.0})
        cost = (resp.input_tokens / 1000 * pricing["input"]) + \
               (resp.output_tokens / 1000 * pricing["output"])

        retrieval_sim_p50 = statistics.median(c.similarity for c in chunks)
        retrieved_context = "\n\n".join(c.text for c in chunks)

        log.info(
            "sut_response",
            model=resp.model,
            latency_ms=resp.latency_ms,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            estimated_cost_usd=round(cost, 6),
            retrieval_sim_p50=retrieval_sim_p50,
        )

        return InstrumentedResponse(
            answer=resp.answer,
            model=resp.model,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency_ms=resp.latency_ms,
            estimated_cost_usd=round(cost, 6),
            retrieved_context=retrieved_context,
            retrieval_sim_p50=retrieval_sim_p50,
        )
