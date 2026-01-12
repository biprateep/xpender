"""
experimental script to use mpi4py instead of dask to generate cutouts and store them in a zarr file.
In an interactive node:
salloc -N 4 -n 512 -t 240 -C cpu -q interactive --image=biprateep/desi-dask:latest --account=m4236
srun -n 512 shifter --image=biprateep/desi-dask:latest  python create_cutouts_mpi.py

Alternatively Slurm file is provided in the same directory.
"""

import os
import sys
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from mpi4py import MPI
from map.views import get_layer

# ================= CONFIGURATION =================
# Zarr Chunking: (Time/Object, Band, H, W)
# Using 100 objects per chunk is a good balance for I/O
ZARR_CHUNK_OBJECTS = 100
BATCH_SIZE = ZARR_CHUNK_OBJECTS  # Number of images to process before writing to disk
IMG_H, IMG_W = 128, 128
N_BANDS = 6  # g, r, i, z, w1, w2

# Paths (Replicating your notebook logic)
release = "iron"
SCRATCH = os.environ.get("SCRATCH", ".")
DEST_PATH = Path(SCRATCH) / "data" / "foundation" / f"{release}"
INPUT_CATALOG = DEST_PATH / "desi_img_params_maglim_19_5.parquet"
OUTPUT_ZARR = DEST_PATH / "desi_maglim_19_5.zarr"

# OUTPUT_ZARR = DEST_PATH / "test.zarr" # TEMP for testing


# ================= HELPER FUNCTION =================
def process_batch(df_batch):
    """
    Process a batch of rows (dataframe slice) and return
    the stacked image and ivar arrays.
    """
    batch_images = []
    batch_ivars = []

    # Use itertuples for faster iteration than iterrows
    for p in df_batch.itertuples(index=False):

        # --- Logic copied and adapted from your notebook ---
        tempfiles = list()
        output = None

        # 1. LS DR9 (g, r, z)
        if p.lsdr9_overlap_flag:
            layer = get_layer(p.ls9_layer_name)
            try:
                img_grz, ivar_grz, hdr_grz = layer.write_cutout(
                    p.ra,
                    p.dec,
                    p.pixscale,
                    p.width,
                    p.height,
                    output,
                    bands="grz",
                    fits=True,
                    jpeg=False,
                    tempfiles=tempfiles,
                    get_images=True,
                    with_invvar=True,
                )
            except Exception:
                img_grz = [np.zeros((p.width, p.height)) for _ in range(3)]
                ivar_grz = [np.zeros((p.width, p.height)) for _ in range(3)]
        else:
            img_grz = [np.zeros((p.width, p.height)) for _ in range(3)]
            ivar_grz = [np.zeros((p.width, p.height)) for _ in range(3)]

        # 2. UnWISE (w1, w2)
        if p.wise_overlap_flag:
            layer = get_layer(p.wise_layer_name)
            try:
                img_w1w2, ivar_w1w2, hdr_w1w2 = layer.write_cutout(
                    p.ra,
                    p.dec,
                    p.pixscale,
                    p.width,
                    p.height,
                    output,
                    bands="12",
                    fits=True,
                    jpeg=False,
                    tempfiles=tempfiles,
                    get_images=True,
                    with_invvar=True,
                )
            except Exception:
                img_w1w2 = [np.zeros((p.width, p.height)) for _ in range(2)]
                ivar_w1w2 = [np.zeros((p.width, p.height)) for _ in range(2)]
        else:
            img_w1w2 = [np.zeros((p.width, p.height)) for _ in range(2)]
            ivar_w1w2 = [np.zeros((p.width, p.height)) for _ in range(2)]

        # 3. LS DR10 (i)
        if p.lsdr10_overlap_flag:
            layer = get_layer(p.ls10_layer_name)
            try:
                img_i, ivar_i, hdr_i = layer.write_cutout(
                    p.ra,
                    p.dec,
                    p.pixscale,
                    p.width,
                    p.height,
                    output,
                    bands="i",
                    fits=True,
                    jpeg=False,
                    tempfiles=tempfiles,
                    get_images=True,
                    with_invvar=True,
                )
            except Exception:
                img_i = [np.zeros((p.width, p.height))]
                ivar_i = [np.zeros((p.width, p.height))]
        else:
            img_i = [np.zeros((p.width, p.height))]
            ivar_i = [np.zeros((p.width, p.height))]

        # Assemble bands: [g, r, z] + insert [i] at 2 -> [g, r, i, z] + [w1, w2]
        # Result: g, r, i, z, w1, w2
        img_grz.insert(2, *img_i)
        img_grz.extend(img_w1w2)

        ivar_grz.insert(2, *ivar_i)
        ivar_grz.extend(ivar_w1w2)

        batch_images.append(np.array(img_grz))
        batch_ivars.append(np.array(ivar_grz))

    # Convert to numpy arrays
    # Clip to prevent overflow during float16 casting (as per notebook)
    images_np = np.clip(batch_images, -65500, 65500).astype(np.float16)
    ivars_np = np.clip(batch_ivars, -65500, 65500).astype(np.float16)

    return images_np, ivars_np


# ================= MAIN EXECUTION =================
def run():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- 1. Load Data ---
    # Every rank reads the catalog. It's usually small enough (metadata only).
    # If this fails, Rank 0 can read and broadcast, but direct read is simpler.
    if rank == 0:
        print(f"Reading catalog from {INPUT_CATALOG}...")

    # Reading catalog
    try:
        df = pd.read_parquet(INPUT_CATALOG)
        # Ensure consistent order across all ranks
        df = df.sort_index()
        # TEMP: limit to 1000 for testing (comment out for full run)
        # df = df.iloc[0:1000]
    except Exception as e:
        print(f"Rank {rank} failed to read catalog: {e}")
        comm.Abort(1)
        return

    total_objects = len(df)

    # --- 2. Initialize Zarr (Rank 0 only) ---
    if rank == 0:
        print(f"Creating Zarr store at {OUTPUT_ZARR}")
        print(f"Total objects: {total_objects}")
        print(f"Chunk size: ({ZARR_CHUNK_OBJECTS}, {N_BANDS}, {IMG_H}, {IMG_W})")

        root = zarr.open_group(str(OUTPUT_ZARR), mode="w")

        # Create arrays with specified chunk sizes
        # Using float16 ('f2') as requested
        root.create(
            "IMG",
            shape=(total_objects, N_BANDS, IMG_H, IMG_W),
            chunks=(ZARR_CHUNK_OBJECTS, N_BANDS, IMG_H, IMG_W),
            dtype="f2",
            compressor=None,
        )

        root.create(
            "IMG_IVAR",
            shape=(total_objects, N_BANDS, IMG_H, IMG_W),
            chunks=(ZARR_CHUNK_OBJECTS, N_BANDS, IMG_H, IMG_W),
            dtype="f2",
            compressor=None,
        )

    # Wait for Rank 0 to finish creating the file structure
    comm.Barrier()

    # --- 3. Distribute Work ---
    # Open Zarr in read/write mode
    root = zarr.open_group(str(OUTPUT_ZARR), mode="r+")
    img_ds = root["IMG"]
    ivar_ds = root["IMG_IVAR"]

    # Split indices among ranks
    all_indices = np.arange(total_objects)
    my_indices = np.array_split(all_indices, size)[rank]

    num_my_tasks = len(my_indices)
    if num_my_tasks == 0:
        print(f"Rank {rank}: No work assigned.")
        return

    print(
        f"Rank {rank}: Processing {num_my_tasks} objects ({my_indices[0]} to {my_indices[-1]})"
    )

    # --- 4. Processing Loop ---
    # Iterate in batches to reduce I/O calls
    for i in range(0, num_my_tasks, BATCH_SIZE):
        batch_idx = my_indices[i : i + BATCH_SIZE]

        # Get the DataFrame slice for this batch
        # iloc is slow for random access, but here indices are sequential per rank
        # Optimization: df is already sorted and split, so we can slice directly?
        # Safe way: use df.iloc with the integer positions from my_indices
        batch_df = df.iloc[batch_idx]

        # Get Images
        try:
            imgs, ivars = process_batch(batch_df)

            # Write to Zarr
            # Note: batch_idx is a numpy array of integers.
            # Zarr supports coordinate selection or slice.
            # Since my_indices are contiguous from array_split, we can use slices.
            start = batch_idx[0]
            stop = batch_idx[-1] + 1
            file_slice = slice(start, stop)

            img_ds[file_slice] = imgs
            ivar_ds[file_slice] = ivars

        except Exception as e:
            print(f"Rank {rank} ERROR on batch starting at {batch_idx[0]}: {e}")
            # Continue to next batch instead of crashing
            continue

        # Optional: Print progress every few batches
        if i % (BATCH_SIZE * 10) == 0 and rank % 10 == 0:
            print(f"Rank {rank}: Done {i}/{num_my_tasks}")

    # --- 5. Cleanup ---
    comm.Barrier()
    if rank == 0:
        print("All ranks finished.")


if __name__ == "__main__":
    run()
