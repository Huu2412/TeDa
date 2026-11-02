# TeDA

Text-Guided and Dialect-Aware Representation Learning for Vietnamese ASR with Audio-Only Inference.

ViP-VL (ChunkFormer encoder) fine-tuned with CTC on ViMD, with a 3-region accent head and a
PhoBERT-guided alignment loss that are used at training time only (inference is audio-only).

**Status:** step 0 (data preparation) and step 1 (encoder + CTC baseline E00, freeze search) are
implemented. Accent head (E01), CIF alignment (E02/E03) and pseudo-labelling (E04) come next.
A Vietnamese walk-through of every folder and file is in [`docs/ARCHITECTURE_VI.md`](docs/ARCHITECTURE_VI.md).

## Quick start on Kaggle (2x T4, Internet on)

`notebooks/kaggle_run.ipynb` runs everything below in order. In a terminal:

```bash
git clone <your repo> && cd teda && pip install -e .

# step 0 - once; keep wav16k/ as a PRIVATE Kaggle Dataset (ViMD is CC-BY-NC-ND-4.0)
python scripts/prepare_data.py --out-dir /kaggle/working/vimd16k --workers 4
python scripts/scan_text.py --manifest-raw /kaggle/working/vimd16k/manifest_raw.tsv \
    --out-dir outputs/scan --manifest-out /kaggle/working/vimd16k/manifest.tsv
python scripts/build_tokenizer.py --data-dir /kaggle/working/vimd16k        # commit artifacts/tokenizer/

# step 1 - check before training
python scripts/check_encoder.py                      # real ViP-VL bundle loads into the vendored encoder
python scripts/probe_memory.py --frames 12000 18000 24000 32000    # pick train.max_frames_per_batch (1 GPU)

# train: re-running the same command resumes; a finished run is a no-op
# MALLOC_ARENA_MAX=2 limits glibc arena count -> reduces host RAM fragmentation on Kaggle
# train.pad_multiple=128 rounds each batch's padded T-dim to a multiple of 128 frames;
#   cuDNN caches one plan per unique tensor shape, so fewer shapes = less host RAM consumed.
# train.max_host_gib=22 saves last.pt and exits if host RSS exceeds 22 GiB; re-run to resume.
MALLOC_ARENA_MAX=2 torchrun --nproc_per_node=2 scripts/train.py --exp e00_baseline \
    --set data.data_dir=/kaggle/working/vimd16k \
           train.pad_multiple=128 train.max_host_gib=22
python scripts/sweep.py --set data.data_dir=/kaggle/working/vimd16k   # freeze search -> outputs/sweep_freeze/summary.tsv
python scripts/evaluate.py --exp e00_baseline --split valid --set data.data_dir=/kaggle/working/vimd16k
```

`--set key=value` overrides any config value; a misspelt key is an error (use `+key=value` to add a new one).
The test split is only evaluated with `--final`.

## Tests

```bash
pip install -e . pytest && pytest        # CPU only, about 2 minutes
```

CPU tests cover the data pipeline, text normalisation, metrics, config handling, freezing, checkpoint/resume,
a 2-process DDP run (gloo), `evaluate.py` and `sweep.py`. **Not covered (needs a GPU): fp16 autocast with
GradScaler, NCCL, and real ViP-VL weights** - `check_encoder.py` and `probe_memory.py` are the first checks to run.

## Credits

`third_party/chunkformer/` vendors the ChunkFormer encoder code (CC BY 4.0, commit 24b1258, unmodified). ViP-VL
weights are downloaded from `khanhld/vip-vl-base-vie`. Please cite the ChunkFormer paper (ICASSP 2025).

## E02 / E03: word-level alignment to PhoBERT (training only)

```bash
# 1) once: PhoBERT word vectors for every utterance (units = the words of text_norm, so CIF fires once per word)
python scripts/build_phobert_cache.py --data-dir $DATA_DIR --out /kaggle/working/phobert_cache --unit syllable
#    --unit word  = VnCoreNLP word segmentation (what PhoBERT was pretrained on; needs `pip install py_vncorenlp` + a JDK)
# 2) E02 = E00 + alignment loss (accent off).  E03 = E00 + accent + alignment.
torchrun --nproc_per_node=2 scripts/train.py --exp e02_full_align \
    --set data.data_dir=$DATA_DIR data.align_cache=/kaggle/working/phobert_cache train.max_hours=10
# controls (does PhoBERT matter, or is any auxiliary loss enough?):
#    model.align.target_mode=random_word   |   model.align.target_mode=shuffle
```

The CIF aligner turns encoder frames into one vector per word, projects it and compares it (cosine) with the frozen,
standardised PhoBERT vector of that word. A count loss teaches CIF to fire N times for N words. Before
`model.align.start_step` the aligner trains on detached features, then both weights ramp up. Validation logs
`align cos`, `align top1` (does word j of the audio retrieve word j of the text among the sentence's words?) and `qty err`.
Inference uses only the encoder and the CTC head.
