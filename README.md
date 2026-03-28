# Person Swap Pipeline

Full-body AI person swap pipeline using open-source models.

## Stack
- SAM 2 — body segmentation
- DWPose — pose estimation  
- OOTDiffusion — body appearance transfer
- InstantID — face identity swap
- HairFastGAN — hair transfer
- ProPainter — temporal consistency
- Real-ESRGAN — 4x upscale

## Structure
- pipeline/ — all AI model scripts
- app/ — Gradio app for HuggingFace Spaces
- frontend/ — Next.js website