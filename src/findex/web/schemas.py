"""Моделі Pydantic. Вони стоять лише на вході й виході API; всередині лишаються dataclass-и
з лаб 2-3 (Index, Posting, SearchResult). Pydantic перевіряє дані з мережі, а свої структури
в пам'яті вже перевірені, тож платити за валідацію на кожне поле всередині не треба."""

from pydantic import BaseModel, Field


class ResultItem(BaseModel):
    doc_id: int
    title: str
    score: float
    path: str
    snippet: str | None = Field(None, description="уривок тексту навколо збігу")
    highlights: list[tuple[int, int]] = Field(
        [], description="де в snippet стоять терми запиту: [початок, кінець)"
    )  # pydantic копіює типове значення для кожного екземпляра


class SearchResponse(BaseModel):
    query: str
    scorer: str
    total: int = Field(description="скільки документів відповідає запиту")
    page: int
    pages: int
    k: int = Field(description="результатів на сторінці")
    took_ms: float
    results: list[ResultItem]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    tokens: int
    text: str
    truncated: bool


class StatsResponse(BaseModel):
    documents: int
    terms: int
    postings: int
    avg_doc_length: float
    storage: str
    index_file_mb: float | None
    uptime_seconds: float
    version: str


class HealthResponse(BaseModel):
    status: str
    documents: int


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
