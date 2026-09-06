from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from sut.retriever import Retriever
from sut.instrumentation import InstrumentedGenerator, InstrumentedResponse

load_dotenv()

app = FastAPI(title="EvalHarness SUT")

retriever = Retriever()
generator = InstrumentedGenerator()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

    def model_post_init(self, _context: object) -> None:
        if len(self.question) > 2000:
            raise ValueError("question must be <= 2000 characters")
        if not 1 <= self.top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")


@app.post("/query")
def query(req: QueryRequest) -> dict:
    chunks = retriever.retrieve(req.question, top_k=req.top_k)
    resp: InstrumentedResponse = generator.run(req.question, chunks)
    return {
        "answer": resp.answer,
        "model": resp.model,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "latency_ms": resp.latency_ms,
        "estimated_cost_usd": resp.estimated_cost_usd,
        "retrieval_sim_p50": resp.retrieval_sim_p50,
        "retrieved_context": resp.retrieved_context,
    }
