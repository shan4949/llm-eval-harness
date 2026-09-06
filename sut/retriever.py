import chromadb
from pathlib import Path
from dataclasses import dataclass

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

@dataclass
class RetrievedChunk:
    text: str
    source: str        # which file it came from
    similarity: float  # how relevant it is (0-1)

class Retriever:
    def __init__(self):
        # create a chromadb client that saves to disk
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection("fomc_statements")

    def _chunk_text(self, text: str, chunk_size: int = 500) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def index(self):
        documents, ids, metadatas = [], [], []

        for file in DATA_DIR.glob("*.txt"):
            text = file.read_text(encoding="utf-8")
            chunks = self._chunk_text(text)

            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                ids.append(f"{file.stem}_chunk_{i}")
                metadatas.append({"source": file.name})

        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Indexed {len(documents)} chunks from {len(list(DATA_DIR.glob('*.txt')))} files")

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k
        )
        return [
            RetrievedChunk(
                text=text,
                source=meta["source"],
                similarity=round(1 - distance, 4),
            )
            for text, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]