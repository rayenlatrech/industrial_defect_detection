# Industrial Casting Defect Detection

Automated visual inspection for manufactured metal castings. A CNN classifies each
part as **OK** or **defective**, an anomaly detector scores how far the part is from
known-good parts, Grad-CAM shows *where* the model is looking, and a report generator
turns all of it into a written inspection report. Everything is wrapped in a
Streamlit app where you upload a photo of a part and get a decision.

## Pipeline

```
image ──► CNN classifier ──► OK / DEFECT + confidence
              │
              ├──► penultimate-layer features ──► kNN vs. OK parts ──► anomaly score
              │
              └──► Grad-CAM ──► heatmap of the region driving the decision
                                        │
                   all three ──► rule-based inspection report
```

## Dataset

[Casting product image data for quality inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
(Kaggle): about 7,300 grayscale top-view photos of submersible pump impellers,
labeled `ok_front` or `def_front` (blow holes, burrs, shrinkage, surface defects),
with an official train/test split.

## Components

**1. CNN classifier** (`models/cnn_classifier.py`, `train_classifier.py`)
A 4-block CNN (Conv → BatchNorm → ReLU → MaxPool, 32 to 256 channels) with a
fully connected head, trained from scratch on 224×224 inputs. Training uses random
crops, flips, rotations and brightness/contrast jitter; 20% of the training folder is
held out for validation and the best checkpoint is kept.

**2. Anomaly detection** (`anomaly_detection.py`)
Images are embedded with the CNN's penultimate layer. Three detectors are fitted on
**OK parts only** and compared on the test set (ROC-AUC and average precision):
- kNN: mean distance to the 5 nearest OK parts (used in the app)
- Isolation Forest
- One-Class SVM

**3. Explainability** (`utils/grad_cam.py`, `grad_cam_demo.py`)
Grad-CAM on the last convolutional layer, overlaid on the input image, so an
operator can check that a rejection is driven by the actual defect and not by the
background.

**4. Inspection report** (`report_generator.py`)
Combines the predicted class, model confidence and anomaly risk level into a
plain-language report with recommended actions. When the classifier and the anomaly
detector disagree, the report flags the part for manual review.

**5. Streamlit app** (`app.py`)
Upload an image and see the prediction, confidence, Grad-CAM heatmap, anomaly score
and generated report side by side.

## How to run

```bash
pip install -r requirements.txt
```

Download the dataset and place it so the folders look like this:

```
data/raw/casting_data/
├── train/
│   ├── def_front/
│   └── ok_front/
└── test/
    ├── def_front/
    └── ok_front/
```

Then, from the project root:

```bash
python train_classifier.py     # trains the CNN, saves checkpoints/simple_cnn_best.pth
python evaluate.py             # accuracy, confusion matrix, classification report on test
python anomaly_detection.py    # compares kNN / Isolation Forest / One-Class SVM
python grad_cam_demo.py        # Grad-CAM overlays on test images
streamlit run app.py           # interactive inspection app
```

## Limitations

- The anomaly detectors use features from a classifier that was trained on labeled
  defects, so they are not fully unsupervised. A stricter setup would embed images
  with a model that never saw a defect (for example an ImageNet backbone, as in PatchCore).
- The anomaly risk thresholds in the report are hand-set for kNN distances and would
  need recalibration on new data.
- All images come from one product and one camera setup.

## Project structure

```
├── models/cnn_classifier.py   # CNN architecture
├── utils/dataset.py           # folder-based PyTorch dataset
├── utils/grad_cam.py          # Grad-CAM implementation
├── train_classifier.py        # training loop
├── evaluate.py                # test-set evaluation
├── anomaly_detection.py       # kNN / Isolation Forest / One-Class SVM
├── grad_cam_demo.py           # heatmap visualisation
├── report_generator.py        # rule-based inspection report
└── app.py                     # Streamlit app
```

## Author

Rayen Latrech · [GitHub](https://github.com/rayenlatrech)
