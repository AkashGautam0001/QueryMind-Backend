import re
from dataclasses import dataclass

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_WORD_PATTERN = re.compile(r"[a-z0-9]+")

# Below this fraction of shared significant words between a citing sentence
# and its cited source, the citation is considered unverified. This is a
# cheap lexical-overlap heuristic, not semantic similarity — it catches the
# common failure mode (citing the wrong source entirely) without the cost
# or complexity of an embedding comparison at generation time.
_MIN_OVERLAP_RATIO = 0.15

# Words too common to be meaningful signal for overlap checking.
_STOPWORDS = frozenset(
    "a an the is are was were be been being this that these those it its of in on "
    "for to with as by at from and or but not no so if then than there here which "
    "who what when where why how do does did has have had will would could should".split()
)


@dataclass
class VerifiedCitation:
    index: int
    chunk_id: object
    document_id: object
    filename: str
    page_number: int | None
    verified: bool


def _significant_words(text: str) -> set[str]:
    return {w for w in _WORD_PATTERN.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _overlap_ratio(sentence: str, source_content: str) -> float:
    sentence_words = _significant_words(sentence)
    if not sentence_words:
        return 0.0
    source_words = _significant_words(source_content)
    shared = sentence_words & source_words
    return len(shared) / len(sentence_words)


def extract_cited_indices(answer_text: str) -> set[int]:
    return {int(m) for m in _CITATION_PATTERN.findall(answer_text)}


def verify_citations(
    answer_text: str, citations_by_index: dict[int, object]
) -> tuple[str, list[VerifiedCitation]]:
    """Checks every [n] citation in the generated answer against the
    source it claims to cite. Citations to an index that doesn't exist in
    the assembled context are stripped from the text outright — that's
    the LLM referencing a source that was never given to it. Citations to
    a real index but with weak lexical overlap between the citing
    sentence and the source content are flagged as unverified (kept in
    the text, but excluded from the trusted citations list returned to
    the caller) rather than silently trusted.
    """
    sentences = _SENTENCE_SPLIT_PATTERN.split(answer_text)
    verified: dict[int, VerifiedCitation] = {}
    cleaned_text = answer_text

    for sentence in sentences:
        for match in _CITATION_PATTERN.finditer(sentence):
            idx = int(match.group(1))
            citation = citations_by_index.get(idx)
            if citation is None:
                # Cited an index that was never in the assembled context —
                # strip the marker; it's pointing at nothing.
                cleaned_text = cleaned_text.replace(match.group(0), "", 1)
                continue

            sentence_without_markers = _CITATION_PATTERN.sub("", sentence)
            ratio = _overlap_ratio(sentence_without_markers, citation.content)
            is_verified = ratio >= _MIN_OVERLAP_RATIO
            already_verified = idx in verified and verified[idx].verified

            # A citation index can appear more than once; once verified
            # anywhere, keep it verified (don't let a later weak mention
            # downgrade an earlier well-supported one).
            verified[idx] = VerifiedCitation(
                index=idx,
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                filename=citation.filename,
                page_number=citation.page_number,
                verified=is_verified or already_verified,
            )

    return cleaned_text, sorted(verified.values(), key=lambda c: c.index)
