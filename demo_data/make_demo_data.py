"""Build the committed MIST demo dataset (a small REAL subset of HEST-1k CCRCC).

This script produced the files under `demo_data/` that ship with the repo. It takes two
real CCRCC slides, keeps a contiguous spatial crop of ~N spots from each (so local kNN
neighborhoods stay meaningful), and subsets the expression to the 50-gene HVG panel.

Subsetting the stored h5ad to the panel genes is lossless for this pipeline: `data.py`
already selects the panel genes *before* normalization, so the loaded, normalized
expression is identical whether the h5ad stores 50 genes or the full transcriptome.

The committed output is what a user runs; this script is kept only for provenance and is
not expected to run outside the original environment (it reads the internal HEST paths).

Usage (regenerate):
    python make_demo_data.py --src_root /path/to/dataset --embed_root /path/to/embed_dataroot
"""
import os
import json
import argparse
import numpy as np
import h5py
import scanpy as sc

HERE = os.path.dirname(os.path.abspath(__file__))
ENCODER = "uni_v1_official"
SRC_COHORT = "CCRCC"
# (sample_id, split role) — one train slide, one test slide
SLIDES = [("INT1", "train"), ("INT5", "test")]


def contiguous_crop(coords, n_keep, seed=0):
    """Indices of a contiguous spatial patch: the n_keep spots nearest to a random seed spot."""
    n = len(coords)
    if n <= n_keep:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    center = coords[rng.integers(n)]
    d2 = ((coords - center) ** 2).sum(axis=1)
    return np.sort(np.argsort(d2)[:n_keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", default="../../../dataset",
                    help="HEST source_dataroot (adata/ + gene panels)")
    ap.add_argument("--embed_root", default="../../../embed_dataroot",
                    help="HEST embed_dataroot (feature .h5)")
    ap.add_argument("--n_spots", type=int, default=400)
    args = ap.parse_args()

    out_embed = os.path.join(HERE, "embed", "DEMO", ENCODER, "fp32")
    out_adata = os.path.join(HERE, "source", "DEMO", "adata")
    out_splits = os.path.join(HERE, "source", "DEMO", "splits")
    for d in (out_embed, out_adata, out_splits):
        os.makedirs(d, exist_ok=True)

    genes = json.load(open(os.path.join(args.src_root, SRC_COHORT, "var_50genes.json")))["genes"]
    json.dump({"genes": genes},
              open(os.path.join(HERE, "source", "DEMO", "var_50genes.json"), "w"), indent=2)

    for sid, _role in SLIDES:
        h5_in = os.path.join(args.embed_root, SRC_COHORT, ENCODER, "fp32", f"{sid}.h5")
        with h5py.File(h5_in, "r") as f:
            barcodes = f["barcodes"][:]
            coords = f["coords"][:]
            embeddings = f["embeddings"][:]

        keep = contiguous_crop(coords, args.n_spots)
        barcodes, coords, embeddings = barcodes[keep], coords[keep], embeddings[keep]

        with h5py.File(os.path.join(out_embed, f"{sid}.h5"), "w") as f:
            f.create_dataset("barcodes", data=barcodes)
            f.create_dataset("coords", data=coords)
            f.create_dataset("embeddings", data=embeddings)

        bc_str = [b[0].decode() if isinstance(b[0], bytes) else str(b[0]) for b in barcodes]
        full = sc.read_h5ad(os.path.join(args.src_root, SRC_COHORT, "adata", f"{sid}.h5ad"))
        full = full[bc_str]
        keep_genes = [g for g in genes if g in full.var_names]
        sub = full[:, keep_genes]
        # write a MINIMAL AnnData (raw counts + names only) — drop obsm/uns/raw/layers to keep
        # the committed file small; the loader only needs X, obs_names, var_names.
        X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
        import anndata as adm
        adata = adm.AnnData(X=X.astype(np.float32),
                            obs=sub.obs[[]].copy(), var=sub.var[[]].copy())
        adata.write_h5ad(os.path.join(out_adata, f"{sid}.h5ad"))
        print(f"{sid}: {len(keep)} spots, {len(keep_genes)}/{len(genes)} panel genes present")

    # one fold: INT1 train, INT5 test (columns match the HEST split schema)
    def write_split(path, sid):
        with open(path, "w") as f:
            f.write("sample_id,patches_path,expr_path\n")
            f.write(f"{sid},patches/{sid}.h5,adata/{sid}.h5ad\n")
    write_split(os.path.join(out_splits, "train_0.csv"), "INT1")
    write_split(os.path.join(out_splits, "test_0.csv"), "INT5")
    print("wrote splits train_0.csv / test_0.csv")


if __name__ == "__main__":
    main()
