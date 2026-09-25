import hashlib

def hash_md5_trunc8(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]