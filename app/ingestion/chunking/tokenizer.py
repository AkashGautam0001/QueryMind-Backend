import tiktoken

_encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text : str) -> int:
    return len(_encoding.encode(text))

def encode(text: str) -> list[int]:
    return _encoding.encode(text)

def decode(token : list[int]) -> str:
    return _encoding.decode(token)