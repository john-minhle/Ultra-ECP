# Ultra-ECP

Parameter-efficient fine-tuning of **UltraSAM** for single-point fetal cardiac chamber segmentation on the [FOCUS](https://zenodo.org/) dataset.

**Paper:** Le, M. et al. "Ultra-ECP." *MIDL 2026*.
[Full paper (MLR Proceedings)](https://proceedings.mlr.press/v315/le26a.html) ·
[ML Anthology](https://mlanthology.org/midl/2026/le2026midl-ultraecp/)

Three components (paper):

1. **LoRA** (rank 8) on prompt encoder + mask decoder; frozen image encoder  
2. **Ellipse-Aware Loss** — anatomical ellipse regularization  
3. **Point-Robust Augmentation** — random jitter of point prompts around mask centroid  

## Repository layout

```
ultra-ecp/
├── ultra_ecp.py               # Single CLI entry point (train / infer / eval / …)
├── requirements.txt
├── LICENSE
├── scripts/
│   └── download_checkpoint.py
└── src/
    └── ultra_ecp/
        ├── lora_sam.py
        ├── ellipse_aware_loss.py
        ├── data_loader_ultrasam_finetune.py
        ├── train_ultrasam_lora.py
        ├── inference_ultrasam_lora.py
        ├── evaluate_ultrasam_lora.py
        ├── convert_to_coco_for_ultrasam.py
        └── run_ultrasam_inference.py
```

## Setup

```bash
git clone https://github.com/<YOUR_USER>/Ultra-ECP.git
cd Ultra-ECP
pip install -r requirements.txt
pip install -U openmim
mim install mmengine mmcv mmdet

# Clone UltraSAM next to this repo (or inside it):
# git clone https://github.com/CAMMA-public/UltraSam.git UltraSam

python ultra_ecp.py download-checkpoint
```

Checkpoint path: `UltraSam/UltraSam.pth` (~363 MB).

## Data (FOCUS)

Expected layout under `--data-root` (e.g. `data/processed_segmentation`):

```
data/processed_segmentation/
├── training/images/      training/annfiles_mask/
├── validation/images/    validation/annfiles_mask/
└── testing/images/       testing/annfiles_mask/
```

Masks: `{id}-cardiac.png` or `{id}_cardiac.png` (thoracic: `-thorax` / `_thoracic`).

## Commands

### Train (full Ultra-ECP)

```bash
python ultra_ecp.py train \
  --data-root data/processed_segmentation \
  --class-name cardiac \
  --checkpoint UltraSam/UltraSam.pth \
  --config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_point_refine.py \
  --output-dir checkpoints/ultrasam_lora_full \
  --batch-size 4 \
  --epochs 100 \
  --lr 1e-4 \
  --point-jitter-radius 10 \
  --ellipse-weight 1.0 \
  --early-stop-patience 15
```

### Ablation

```bash
# LoRA only
python ultra_ecp.py train ... --output-dir checkpoints/lora_only --no-point-jitter --no-ellipse-loss

# LoRA + point robustness
python ultra_ecp.py train ... --no-ellipse-loss

# LoRA + ellipse loss
python ultra_ecp.py train ... --no-point-jitter
```

### Inference

```bash
python ultra_ecp.py infer \
  --checkpoint checkpoints/ultrasam_lora_full/best.pth \
  --config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_point_refine.py \
  --data-root data/processed_segmentation \
  --split testing \
  --output-dir predictions/testing \
  --class-name cardiac \
  --use-centroid
```

### Evaluation (DSC, HD95, robustness at 0/2/5/10 px)

```bash
python ultra_ecp.py eval \
  --checkpoint checkpoints/ultrasam_lora_full/best.pth \
  --config UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_point_refine.py \
  --data-root data/processed_segmentation \
  --split testing \
  --class-name cardiac \
  --jitter-radii 0 2 5 10 \
  --output results/eval.json
```

### COCO conversion (zero-shot UltraSAM)

```bash
python ultra_ecp.py convert-coco --data-root . --split testing --output-dir ultrasam_data
```

Do **not** commit `UltraSam/`, checkpoints, or raw FOCUS images (see `.gitignore`).

## Note

Training uses MMDetection `init_detector` and a forward helper that may need adjustment for your exact UltraSAM build. See comments in `src/ultra_ecp/train_ultrasam_lora.py`.

## License

This code is released under the [MIT License](LICENSE). It depends on UltraSAM, SAM, and the FOCUS dataset (CC-BY 4.0) — follow their respective licenses when using pretrained weights or data.

## Citation

```bibtex
@inproceedings{le2026ultra,
  title={Ultra-ECP: Ellipse-Constrained and Point-Robust Foundation Model Adaptation for Fetal Cardiac Ultrasound Segmentation},
  author={Le, Minh HN and Le, Khanh TQ and Vinh, Tuan and Nguyen, Thanh-Huy and Huynh, Han H and Pham, Khoa D and Vu, Anh Mai and Kha, Hien Quang and Nguyen, Phat Ky and Bagci, Ulas and others},
  booktitle={Medical Imaging with Deep Learning},
  year={2026}
}
```

## Author

Minh Le

## Affiliation

[TMU AIBioMed Lab](https://www.aibiomedlab.com/member/), In-Service Master Program in Artificial Intelligence in Medicine, Taipei Medical University, Taipei, Taiwan.
