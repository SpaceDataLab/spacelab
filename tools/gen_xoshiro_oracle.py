#!/usr/bin/env python3
"""The splitmix64-seeded xoshiro streams behind two generator tests.

`radiative_transfer.rs` scrambles with **, `spacelab-estimation`'s `sampling.rs` with ++; the linear
state update underneath is the same, which is exactly why each needs its own fixture.

Written from the published algorithm -- Blackman & Vigna, "Scrambled Linear Pseudorandom Number
Generators", ACM TOMS 47(4) (2021), and the authors' public-domain C at https://prng.di.unimi.it/
(splitmix64.c, xoshiro256starstar.c, xoshiro256plusplus.c) -- rather than from the Rust it checks.
Everything is 64-bit integer arithmetic, so the stream is bit-identical in any language that wraps at
2**64; nothing here depends on a libm.

Monte Carlo radiative transfer reports fractions with a statistical error of its own, so a degraded
generator hides inside the tolerance: a wrong shift or a state update that ORs instead of XORs still
produces random-looking numbers, and the reflectance moves by less than the sampling noise. The
stream is the only thing about the generator that is exactly specified, so that is what the test
pins.

`emit` prints the fixture as Rust; the test embeds the words and re-derives nothing.
"""

from __future__ import annotations

M = (1 << 64) - 1

SEEDS = (0, 1, 42)
DRAWS = 4


def rotl(x: int, k: int) -> int:
    """Rotate a 64-bit word left by k bits."""
    return ((x << k) | (x >> (64 - k))) & M


def splitmix64(seed: int, count: int) -> list[int]:
    """The first `count` outputs of splitmix64 from `seed` -- xoshiro's recommended seeder."""
    out, z = [], seed
    for _ in range(count):
        z = (z + 0x9E3779B97F4A7C15) & M
        x = z
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M
        out.append(x ^ (x >> 31))
    return out


class Xoshiro256StarStar:
    """xoshiro256**: 256 bits of state, a `* 5, rotl 7, * 9` scrambler over a linear update."""

    def __init__(self, seed: int) -> None:
        self.s = splitmix64(seed, 4)

    def next_u64(self) -> int:
        s = self.s
        result = (rotl((s[1] * 5) & M, 7) * 9) & M
        t = (s[1] << 17) & M
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 45)
        return result


class Xoshiro256PlusPlus:
    """xoshiro256++: the same linear update, scrambled by `rotl(s0 + s3, 23) + s0`."""

    def __init__(self, seed: int) -> None:
        self.s = splitmix64(seed, 4)

    def next_u64(self) -> int:
        s = self.s
        result = (rotl((s[0] + s[3]) & M, 23) + s[0]) & M
        t = (s[1] << 17) & M
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 45)
        return result


def emit() -> None:
    """Print both fixtures as the Rust arrays the tests embed."""
    for name, generator in (
        ("xoshiro256** (radiative_transfer.rs)", Xoshiro256StarStar),
        ("xoshiro256++ (sampling.rs)", Xoshiro256PlusPlus),
    ):
        print(f"// {name}")
        for seed in SEEDS:
            rng = generator(seed)
            words = ", ".join(f"0x{rng.next_u64():016x}" for _ in range(DRAWS))
            print(f"        ({seed}, [{words}]),")


if __name__ == "__main__":
    emit()
