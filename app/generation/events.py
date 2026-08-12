from dataclasses import dataclass, field

@dataclass
class TokenEvent:
    text: str

@dataclass
class CitationEvent:
    citations: list[dict] = field(default_factory=list)

@dataclass
class DoneEvent:
    message_id: str | None = None
    conversation_id : str | None = None

@dataclass
class NoContextEvent:
    message: str