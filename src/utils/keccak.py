"""
Minimal pure-Python Keccak-256 (Ethereum "keccak256") implementation.

We use this to compute event topic0 hashes (keccak256("EventSig(...)")) without adding new deps.

Note: This is Keccak (not NIST SHA3). Ethereum uses Keccak-256.
"""

from __future__ import annotations

from typing import List

_MASK64 = (1 << 64) - 1


def _rotl64(x: int, n: int) -> int:
    n = int(n) & 63
    return ((x << n) | (x >> (64 - n))) & _MASK64


# Rotation offsets r[x][y]
_R = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

# Round constants
_RC = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
]


def _keccak_f1600(state: List[int]) -> None:
    """
    In-place Keccak-f[1600] permutation.
    state is 25 lanes of 64-bit (little-endian lane order) in a flat list:
      lane(x,y) is at index x + 5*y
    """
    assert len(state) == 25

    for rc in _RC:
        # --- theta ---
        c = [0] * 5
        for x in range(5):
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20]
        d = [0] * 5
        for x in range(5):
            d[x] = c[(x - 1) % 5] ^ _rotl64(c[(x + 1) % 5], 1)
        for y in range(5):
            base = 5 * y
            state[base + 0] ^= d[0]
            state[base + 1] ^= d[1]
            state[base + 2] ^= d[2]
            state[base + 3] ^= d[3]
            state[base + 4] ^= d[4]

        # --- rho + pi ---
        b = [0] * 25
        for x in range(5):
            for y in range(5):
                idx = x + 5 * y
                v = _rotl64(state[idx], _R[x][y])
                nx = y
                ny = (2 * x + 3 * y) % 5
                b[nx + 5 * ny] = v

        # --- chi ---
        for y in range(5):
            base = 5 * y
            b0, b1, b2, b3, b4 = b[base : base + 5]
            state[base + 0] = b0 ^ ((~b1) & b2) & _MASK64
            state[base + 1] = b1 ^ ((~b2) & b3) & _MASK64
            state[base + 2] = b2 ^ ((~b3) & b4) & _MASK64
            state[base + 3] = b3 ^ ((~b4) & b0) & _MASK64
            state[base + 4] = b4 ^ ((~b0) & b1) & _MASK64

        # --- iota ---
        state[0] ^= rc


def keccak_256(data: bytes) -> bytes:
    """
    Compute Keccak-256 digest of data.
    """
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)

    rate = 136  # 1088 bits
    out_len = 32
    state = [0] * 25

    # Absorb full blocks
    off = 0
    while off + rate <= len(data):
        block = data[off : off + rate]
        for i in range(0, rate, 8):
            lane = int.from_bytes(block[i : i + 8], "little")
            state[i // 8] ^= lane
        _keccak_f1600(state)
        off += rate

    # Final block + pad10*1 (Keccak padding, domain 0x01)
    last = bytearray(data[off:])
    last.append(0x01)
    while len(last) % rate != rate - 1:
        last.append(0x00)
    last.append(0x80)

    for i in range(0, rate, 8):
        lane = int.from_bytes(last[i : i + 8], "little")
        state[i // 8] ^= lane
    _keccak_f1600(state)

    # Squeeze
    out = bytearray()
    while len(out) < out_len:
        for i in range(0, rate, 8):
            out.extend(int(state[i // 8] & _MASK64).to_bytes(8, "little"))
            if len(out) >= out_len:
                break
        if len(out) < out_len:
            _keccak_f1600(state)

    return bytes(out[:out_len])


def topic0(event_signature: str) -> str:
    """
    Return 0x-prefixed topic0 hash for an event signature, e.g. "Upgraded(address)".
    """
    sig = str(event_signature).strip()
    return "0x" + keccak_256(sig.encode("utf-8")).hex()

