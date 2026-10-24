# Vendored: ChunkFormer (ViP-VL encoder code)

Source: https://github.com/khanld/chunkformer  (commit 24b12583947d2338bb39f005637ee43a64c364e7)
License: CC BY 4.0 (see LICENSE). Some files also carry WeNet / ESPnet Apache-2.0 headers; kept intact.

Only the files needed to build `ChunkFormerEncoder` are copied, unmodified:

    chunkformer/modules/{attention,convolution,embedding,encoder,encoder_layer,norm,
                         positionwise_feed_forward,subsampling,swish,cmvn}.py
    chunkformer/utils/{class_utils,mask,cmvn}.py

The only change: `chunkformer/__init__.py` is empty (upstream imports the whole toolkit).
`teda/__init__.py` puts this folder on `sys.path`, so `import chunkformer.modules.encoder` works.

Pretrained weights are NOT stored here; they are fetched from the Hugging Face Hub
(`khanhld/vip-vl-base-vie`). Please cite the ChunkFormer paper (ICASSP 2025) in your report.
