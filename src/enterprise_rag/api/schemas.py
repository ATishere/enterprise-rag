from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    tenant_id: str = Field(min_length=1, max_length=255)
    user_id: str = Field(min_length=1, max_length=255)
    stream: bool = False
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    text: str
    chunk_id: str
    confidence: float = Field(ge=0, le=1)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    context_used: int
    hyde_used: bool
    graph_used: bool
    trace_id: str | None = None
