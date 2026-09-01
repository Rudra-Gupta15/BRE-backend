# Deterministic pseudo-random generator seeded from a string, so endpoints
# like inference scoring return the SAME numbers for the same customId/model
# on every call (no DB needed) instead of re-rolling random values.

MASK32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """Mimics JS Math.imul: 32-bit signed integer multiplication."""
    result = (a * b) & MASK32
    if result >= 0x80000000:
        result -= 0x100000000
    return result


def _to_uint32(n: int) -> int:
    return n & MASK32


def _hash_seed(s: str) -> int:
    h = _to_uint32(1779033703 ^ len(s))
    for ch in s:
        h = _imul(h ^ ord(ch), 3432918353) & MASK32
        h = _to_uint32(((h << 13) | (h >> 19)) & MASK32)
    return h


class Rng:
    """mulberry32, seeded the same way as the frontend's utils/rng.js."""

    def __init__(self, seed: str):
        h = _hash_seed(str(seed))
        # advance once, matching rng.js's `let a = seedFn();`
        h = _imul(h ^ (h >> 16), 2246822507) & MASK32
        h = _imul(h ^ (h >> 13), 3266489909) & MASK32
        h = (h ^ (h >> 16)) & MASK32
        self.a = h

    def next(self) -> float:
        self.a = _to_uint32(self.a + 0x6D2B79F5)
        t = self.a
        t = _imul(t ^ (t >> 15), (1 | t))
        t = _to_uint32(t)
        t = (t + _imul(t ^ (t >> 7), (61 | t))) ^ t
        t = _to_uint32(t)
        return _to_uint32(t ^ (t >> 14)) / 4294967296

    def __call__(self) -> float:
        return self.next()


def create_rng(seed) -> Rng:
    return Rng(seed)


def rand_range(rng: Rng, lo: float, hi: float) -> float:
    return lo + rng() * (hi - lo)


def rand_int(rng: Rng, lo: int, hi: int) -> int:
    return int(rand_range(rng, lo, hi + 1))


def pick(rng: Rng, items: list):
    return items[int(rng() * len(items))]
