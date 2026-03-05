"""Quantize cCRE z-scores from float64 to uint8 bin indices.

Reads the original parquet, computes a global min/max across all sample columns,
maps each z-score to a 0–255 bin index, and writes a compact parquet + sidecar JSON.

Usage:
    python preprocess_uint8.py
"""

import json
import math

import polars as pl

DATAFILE = "/Users/conrad/Projects/vedatonuryilmaz/simplex-scatter/comb_healthy_scatac_tcga_encode.parquet"
OUTPUT_PARQUET = "ccre_zscore_uint8.parquet"
OUTPUT_META = "ccre_zscore_uint8_meta.json"
NON_SAMPLE_COLS = {"cCRE", "class", "rDHS", "chr", "start", "end"}
BATCH_SIZE = 200  # columns per streaming pass (fallback if Polars chokes)


def main():
    # ── Step 1: Discover sample columns ──────────────────────────────────
    schema = pl.scan_parquet(DATAFILE).collect_schema()
    sample_cols = [c for c in schema.names() if c not in NON_SAMPLE_COLS]
    print(f"Found {len(sample_cols)} sample columns")

    # ── Step 2: Compute global min/max (batched streaming passes) ────────
    global_min = math.inf
    global_max = -math.inf

    for i in range(0, len(sample_cols), BATCH_SIZE):
        batch = sample_cols[i : i + BATCH_SIZE]
        min_exprs = [pl.col(c).min().alias(f"min__{c}") for c in batch]
        max_exprs = [pl.col(c).max().alias(f"max__{c}") for c in batch]

        stats = (
            pl.scan_parquet(DATAFILE)
            .select(min_exprs + max_exprs)
            .collect(engine="streaming")
        )

        batch_min = min(stats[f"min__{c}"][0] for c in batch)
        batch_max = max(stats[f"max__{c}"][0] for c in batch)
        global_min = min(global_min, batch_min)
        global_max = max(global_max, batch_max)
        print(f"  Batch {i // BATCH_SIZE + 1}: cols {i}–{i + len(batch) - 1}, "
              f"min={batch_min:.6f}, max={batch_max:.6f}")

    step = (global_max - global_min) / 256
    print(f"\nGlobal min={global_min:.6f}, max={global_max:.6f}, step={step:.6f}")

    # ── Step 3: Quantize and write ───────────────────────────────────────
    cast_exprs = [
        pl.when(pl.col(c).is_nan())
        .then(None)
        .otherwise(
            ((pl.col(c) - global_min) / step)
            .floor()
            .cast(pl.Int32)
            .clip(0, 255)
            .cast(pl.UInt8)
        )
        .alias(c)
        for c in sample_cols
    ]

    print(f"\nWriting {OUTPUT_PARQUET} ...")
    (
        pl.scan_parquet(DATAFILE)
        .select([pl.col("cCRE"), pl.col("class")] + cast_exprs)
        .sink_parquet(OUTPUT_PARQUET, compression="zstd", compression_level=3)
    )
    print("Done.")

    # ── Step 4: Write sidecar metadata ───────────────────────────────────
    meta = {
        "global_min": global_min,
        "global_max": global_max,
        "step": step,
        "n_bins": 256,
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written to {OUTPUT_META}")

    print(f"\n── Hardcode these in app.py config ──")
    print(f"ZSCORE_MIN = {global_min}")
    print(f"ZSCORE_MAX = {global_max}")
    print(f"ZSCORE_STEP = {step}")
    print(f"N_BINS = 256")


if __name__ == "__main__":
    main()
