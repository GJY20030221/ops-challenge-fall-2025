import numpy as np
import pandas as pd
from numba import njit


@njit(nogil=True, fastmath=True)
def _rolling_percentile_rank_no_nan(codes, values, window, symbol_count):
    """Numba kernel for NaN-free data."""
    n = values.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out

    win = 1 if window < 1 else window
    buffers = np.empty((symbol_count, win), dtype=np.float32)
    sizes = np.zeros(symbol_count, dtype=np.int32)
    positions = np.zeros(symbol_count, dtype=np.int32)

    for i in range(n):
        sym = codes[i]
        val = values[i]

        pos = positions[sym]
        buf = buffers[sym]
        buf[pos] = val

        size = sizes[sym]
        if size < win:
            size += 1
            sizes[sym] = size

        pos += 1
        if pos == win:
            pos = 0
        positions[sym] = pos

        valid = size
        start = pos - valid
        if start < 0:
            start += win

        cnt = 0
        idx = start
        for _ in range(valid):
            if buf[idx] <= val:
                cnt += 1
            idx += 1
            if idx == win:
                idx = 0

        out[i] = cnt / valid

    return out


@njit(nogil=True)
def _rolling_percentile_rank_with_nan(codes, values, window, symbol_count):
    """Same as above but preserves pandas NaN comparison semantics."""
    n = values.shape[0]
    out = np.empty(n, dtype=np.float32)
    if n == 0:
        return out

    win = 1 if window < 1 else window
    buffers = np.empty((symbol_count, win), dtype=np.float32)
    sizes = np.zeros(symbol_count, dtype=np.int32)
    positions = np.zeros(symbol_count, dtype=np.int32)

    for i in range(n):
        sym = codes[i]
        val = values[i]

        pos = positions[sym]
        buf = buffers[sym]
        buf[pos] = val

        size = sizes[sym]
        if size < win:
            size += 1
            sizes[sym] = size

        pos += 1
        if pos == win:
            pos = 0
        positions[sym] = pos

        valid = size
        start = pos - valid
        if start < 0:
            start += win

        if np.isnan(val):
            out[i] = 0.0
            continue

        cnt = 0
        idx = start
        for _ in range(valid):
            buf_val = buf[idx]
            if not np.isnan(buf_val) and buf_val <= val:
                cnt += 1
            idx += 1
            if idx == win:
                idx = 0

        out[i] = cnt / valid

    return out


def ops_rolling_rank(input_path: str, window: int = 20) -> np.ndarray:
    win = int(window)
    if win <= 0:
        raise ValueError("window must be a positive integer")

    df = pd.read_parquet(input_path, columns=["symbol", "Close"])
    if df.empty:
        return np.empty((0, 1), dtype=np.float32)

    close_vals = df["Close"].to_numpy(dtype=np.float32, copy=False)
    codes, _ = pd.factorize(df["symbol"], sort=False)
    if (codes < 0).any():
        raise ValueError("symbol column contains missing values")

    codes = codes.astype(np.int32, copy=False)
    symbol_count = int(codes.max()) + 1

    if np.isnan(close_vals).any():
        ranks = _rolling_percentile_rank_with_nan(codes, close_vals, win, symbol_count)
    else:
        ranks = _rolling_percentile_rank_no_nan(codes, close_vals, win, symbol_count)

    return ranks.reshape(-1, 1)
