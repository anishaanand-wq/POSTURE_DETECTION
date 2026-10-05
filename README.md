# Baby Sleep Posture Classification (CNN vs MobileNetV2 vs EfficientNet-B0)

Camera-based classification of infant sleep posture into **`left`**, **`prone`** (stomach) and **`sunpine`** (back/supine),
motivated by Sudden Infant Death Syndrome (SIDS) risk. The project follows the contactless, edge-friendly direction of
Huang et al. (2021) and compares a from-scratch CNN against two pretrained lightweight networks.

> **Research prototype only. Not a medical or safety device.** Do not rely on it to monitor a real infant.

## Base paper and literature
| Role | Paper |
|---|---|
| **Base paper** | Huang et al., *Memory-Efficient AI Algorithm for Infant Sleeping Death Syndrome Detection in Smart Buildings*, AI 2021. https://doi.org/10.3390/ai2040042 |
| Related | Nachet & Stambouli, *A Real Time Object Detection System for Infant Safe Sleep Based on YOLOv5 Algorithm*, EDiS 2022. https://doi.org/10.1109/EDiS57230.2022.9996513 |
| Related | Yuan et al., *A Multi-center Clinical Trial for Camera-based Infant Sleep and Awake Detection in Neonatal Intensive Care Unit*, HealthCom 2023. https://doi.org/10.1109/Healthcom56612.2023.10472347 |

Full review and comparison: [docs/literature_review.md](docs/literature_review.md) | BibTeX: [docs/references.bib](docs/references.bib)

## Dataset
See [DATASET.md](DATASET.md). Images are **not** stored in this repository.

## What was done
1. Trained MobileNetV2 on the dataset's original splits: train acc 0.95 but test acc 0.36, a sign of leakage / source mismatch.
2. Found that the 8,043 training files came from ~1,231 originals (augmented copies), and that valid and test came from differently named sources.
3. Pooled all splits, kept one image per original (1,429 images), embedded them with a pretrained CNN, clustered near-duplicate frames (777 groups) and re-split **by group** with `StratifiedGroupKFold` (0 shared groups): train 899 / valid 197 / test 333.
4. Trained three models on the same split with the same loss and early stopping.
5. Exported the best model to ONNX.

## Methods
- Preprocessing: resize 224x224, ImageNet normalisation. Augmentation: RandomResizedCrop, rotation, colour jitter, RandomErasing. **No horizontal flip** (it would turn `left` into right while keeping the label).
- Models: from-scratch baseline CNN (~0.39M params), MobileNetV2 (2.23M), EfficientNet-B0 (~4.0M).
- Transfer learning in two stages: train the new head with a frozen backbone, then fine-tune the last blocks at a low learning rate.
- Training: cross-entropy with class weights and label smoothing (0.1), AdamW with weight decay, dropout 0.5, early stopping on validation loss.
- Evaluation: accuracy, macro F1, per-class recall (`prone` recall is the safety-relevant metric), confusion matrix, bootstrap 95% interval.

## Results (test split, 333 images)
| Model | Test acc | Macro F1 | Recall left | Recall prone | Recall sunpine |
|---|---|---|---|---|---|
| Baseline CNN | 0.333 | 0.324 | 0.739 | 0.092 | 0.556 |
| MobileNetV2 | 0.541 | 0.512 | 0.511 | 0.536 | 0.622 |
| **EfficientNet-B0** | **0.601** | **0.548** | 0.511 | **0.658** | 0.533 |

Always predicting `prone` would score about 0.59 accuracy on this test set, so macro F1 and per-class recall are the fairer numbers.
Raw table: [results/model_comparison.csv](results/model_comparison.csv).

### Why these numbers are lower than the base paper's ~90%
They are not directly comparable: the base paper uses a binary task, its own 10,240-image day/night dataset and a random 70/20/10 split,
while this project uses 3 classes, ~1,400 unique images and a group-aware split that prevents near-duplicate frames crossing splits.
See the comparison table in the literature review.

## Limitations
- Small dataset (~1,400 unique images); `sunpine` has only 45 test images.
- Some frames show non-sleeping poses (e.g. a crawling baby).
- Single dataset; no cross-camera or cross-infant validation.
- Models are not quantised; the memory-efficiency goal of the base paper is only partly addressed (ONNX export done, int8 quantisation is future work).

## Future work
Post-training / quantisation-aware int8 quantisation; sleep-vs-awake gating before posture classification (see Yuan et al.);
baby/face detection with YOLO-style detectors (see Nachet & Stambouli); more unique, night-vision and multi-camera data; edge deployment on Raspberry Pi.

## Technology stack
Computer vision, deep learning (PyTorch / torchvision), machine learning (transfer learning, clustering), data analysis (leakage audit, bootstrap CI),
cloud GPU training (Kaggle), edge-ready export (ONNX). IoT, fog computing, NLP, LLM, RAG and agentic components are **not** implemented.

## Repository layout
```
README.md  DATASET.md  requirements.txt
notebooks/  notebook-posture-detection.ipynb
docs/       literature_review.md  references.bib
results/    model_comparison.csv  class_names.json
models/     (weights are added after training, see models/README.md)
```

## Run
Open the notebook on Kaggle with **GPU** and **Internet** enabled, attach the dataset, and run top to bottom.
Part A is the initial leaky run (kept for transparency); Parts B and C are the corrected pipeline and final results.
