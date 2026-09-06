import anthropic
import time
from dataclasses import dataclass
from dotenv import load_dotenv
from sut.retriever import RetrievedChunk

load_dotenv()

@dataclass
class GeneratorResponse:
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int

class Generator:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, question: str, context_chunks: list[RetrievedChunk]) -> GeneratorResponse:
        # 1. build context string from chunks
        context = "\n\n".join(chunk.text for chunk in context_chunks)

        # 2. build the prompt
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above. "
            "If the answer is not in the context, say 'I don't know'."
        )
        
        # 3. record start time: time.time()
        start_time = time.time()

        # 4. call the API
        response = self.client.messages.create(model=self.model,max_tokens=512, messages=[{"role": "user", "content": prompt}])

        # 5. calculate latency_ms: int((time.time() - start) * 1000)
        latency_ms = int((time.time() - start_time) * 1000)

        # 6. return GeneratorResponse with all fields filled in
        return GeneratorResponse(
            answer=response.content[0].text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )
