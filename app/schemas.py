from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class Source(BaseModel):
    index: int
    source: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    helpful: bool
