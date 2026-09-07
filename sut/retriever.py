import calendar
import chromadb
from pathlib import Path
from dataclasses import dataclass

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

MONTH_NAMES = {i: calendar.month_name[i] for i in range(1, 13)}


def _date_prefix(filename: str) -> str:
    """fomc_2022_03.txt → 'FOMC statement, March 2022.'"""
    parts = filename.replace(".txt", "").split("_")  # ["fomc", "2022", "03"]
    year, month = parts[1], int(parts[2])
    return f"FOMC statement, {MONTH_NAMES[month]} {year}."


@dataclass
class RetrievedChunk:
    text: str
    source: str        # which file it came from
    similarity: float  # how relevant it is (0-1)


class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection("fomc_statements")

    def _chunk_text(self, text: str, chunk_size: int = 500) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def reindex(self):
        """Drop and rebuild the collection from scratch."""
        self.client.delete_collection("fomc_statements")
        self.collection = self.client.get_or_create_collection("fomc_statements")
        self.index()

    def index(self):
        documents, ids, metadatas = [], [], []

        for file in sorted(DATA_DIR.glob("*.txt")):
            text = file.read_text(encoding="utf-8")
            prefix = _date_prefix(file.name)
            stem_parts = file.stem.split("_")  # ["fomc", "2022", "03"]
            year, month = int(stem_parts[1]), int(stem_parts[2])
            chunks = self._chunk_text(text)

            for i, chunk in enumerate(chunks):
                # prepend date so the embedding model sees temporal context
                documents.append(f"{prefix} {chunk}")
                ids.append(f"{file.stem}_chunk_{i}")
                metadatas.append({"source": file.name, "year": year, "month": month})

        self.collection.add(documents=documents, ids=ids, metadatas=metadatas)
        print(f"Indexed {len(documents)} chunks from {len(list(DATA_DIR.glob('*.txt')))} files")

    def _year_filter(self, question: str) -> dict | None:
        """If the question mentions a year we have data for, filter to that year."""
        for year in range(2022, 2027):
            if str(year) in question:
                return {"year": {"$eq": year}}
        return None

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        where = self._year_filter(question)
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            where=where,
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