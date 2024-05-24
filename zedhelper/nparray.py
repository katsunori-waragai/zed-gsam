"""
numpy.ascontiguousarray()
"""

import numpy as np


def as_float32_array(byte_data) -> np.ndarray:
    return np.frombuffer(byte_data, dtype=np.float32)


def as_uint8_array(byte_data) -> np.ndarray:
    return np.frombuffer(byte_data, dtype=np.uint8)


def as_bytes(arr: np.ndarray):
    return arr.tobytes()


def float32array_to_uint8array(float32array: np.ndarray) -> np.ndarray:
    assert float32array.dtype == np.float32
    shape = list(float32array.shape)
    for s in shape:
        assert s > 0
    byte_data = float32array.tobytes()
    shape.append(4)
    return np.reshape(as_uint8_array(byte_data), shape)


if __name__ == "__main__":
    import math

    a = np.full((3), math.pi, dtype=np.float32)
    print(a)
    byte_data = a.tobytes()
    print(f"{byte_data=}")
