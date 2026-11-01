# Motion-Aware Video Violence Detection

https://github.com/user-attachments/assets/9b952c6d-d436-4149-a13d-f35dd0037386

## Abstract

Detecting violence in crowds is an important task that enables public authorities to rapidly respond to high-risk incidents and safeguard public spaces. However, the scarcity of large-scale and diverse annotated data, combined with the growing resilience of large volumes of data for training and evaluating complex deep learning architectures, often leads to poor generalization in violence detection, especially across different camera perspectives. In this paper, considering the importance of robustness in crowd analysis systems, we propose a two-stream X-CLIP–based architecture, namely MAG-XCLIP, designed for data-efficient learning using the available small-scale domain-specific annotated datasets for violence detection that achieves strong generalization performance. In addition, we introduce a motion-aware gate module that can handle variations arising from different camera perspectives. The experimental evaluation shows that the proposed method can significantly outperform the current state-of-the-art violence detection methods in six cross-dataset evaluations, achieving a maximum percentage improvement of 67.57% in ROC AUC against CrimeNet. Moreover, the introduced motion-aware gate module resulted in a maximum percentage improvement of 24.17% in terms of PR AUC compared to a simple (non-motion-aware) gate module when evaluated on content captured from different camera perspectives. Finally, it is demonstrated that with data-efficient learning, MAG-XCLIP can outperform the current state-of-the-art method trained on large-scale violence detection datasets in three cross-dataset experiments, leading to a maximum percentage increase of 34.78% in ROC AUC.

<img width="1904" height="994" alt="mag_xclip_flowchart" src="https://github.com/user-attachments/assets/6f8d7939-1a89-46d0-948e-0164e1fc4feb" />

## Features

- Frozen X-CLIP video feature extraction with `microsoft/xclip-base-patch32`, improving cross-domain generalization.
- NeuFlowV2 optical flow estimation through PTLFlow.
- Motion-aware gating between appearance and motion representations, improving static to dynamic camera generalization.
- Accuracy, ROC-AUC, PR-AUC, and average precision evaluation.
- Optimal threshold search.
- Video export with violence probability overlays.
- Predefined partitions of violence detection datasets for reproducibility and reusability of the results.

## Project files

| Path | Purpose |
| --- | --- |
| `train.py` | Train MAG-XCLIP and preserve the best performing model based on a validation metric. |
| `evaluate.py` | Evaluate MAG-XCLIP on normal and violence videos using various classification metrics. |
| `optimal_threshold.py` | Search the optimal probability thresholds and display the confusion matrix. |
| `overlay.py` | Export annotated videos without evaluation metrics or ground-truth labels. |
| `splits/` | Text files containing video paths and labels for each partition (train, validation, test). |
| `get_test.py` | Copy videos from a split txt file into a directory for evaluation. |
| `update_paths.py` | Update the video directory paths in the split txt files. |
| `checkpoints/` | Contains the trained MAG-XCLIP models. |
| `annotated_videos/` | Default annotated video output. |


## Setup

Create and activate a Conda environment:

```bash
conda create --name magxclip python=3.11
conda activate magxclip
```

Install PyTorch for your CPU or CUDA environment, then install the remaining packages required by the scripts:

```bash
python -m pip install transformers ptlflow opencv-python numpy Pillow scikit-learn matplotlib seaborn
```


## Quick start: export annotated videos

Download the pretrained MAG-XCLIP models from [google drive](https://drive.google.com/drive/folders/12pYjbj7M7pdvorcY3O-lhraM-jyFG7eb?usp=sharing) and place them inside the `/checkpoints` directory. Then execute the following command:

```bash
# This is an example command. Replace the placeholder paths with your actual paths.
python overlay.py --input-dir ./videos --output-dir ./annotated_videos --model-path ./checkpoints/fusion_head_best_pr_auc.pth --device cuda:0
```

| Violence probability | Border color |
| --- | --- |
| 0% | Green |
| 50% | Yellow |
| 100% | Red |

## Dataset preparation

### Definition of the class labels for training and evaluation

| Label | Class |
| --- | --- |
| `0` | Normal / non-violent |
| `1` | Violence |


### Update existing split paths

The following command updates the paths in every top-level `*.txt` file in the selected dataset directory, including the training, validation, and test files, to match the actual paths on your disk.

```bash
# This is an example command. Replace the placeholder paths with your actual paths.
python splits/update_paths E:\mag-xclip-main\splits\Movies E:\mag-xclip-main\movies_dataset\noFights E:\mag-xclip-main\movies_dataset\fights
```


In detail, the training, validation, and test split `.txt` files contain one video path and a class label per line:

```text
#Example
./movies_dataset/noFights/example_normal.mp4 0
./movies_dataset/fights/example_fight.mp4 1
```

## Training

To start training execute the following command:

```bash
python train.py --train-file ./splits/WVD/train.txt --valid-file ./splits/WVD/valid.txt --device cuda:0 --window-size 8 --batch-size 2 --num-epochs 3 --lr 0.0003 --lambda-adv 0.1 --best-metric pr_auc --checkpoint-dir ./checkpoints
```

Use `pr_auc` for imbalanced datasets and `roc_auc` for balanced. The best performing model will be saved inside the `/checkpoints` directory.


## Evaluation

The evaluation requires a MAG-XCLIP-trained model, a folder containing normal videos, and a folder containing violence ones. Then execute the following command:

```bash
python evaluate.py --normal-dir ./test_videos/normal --violence-dir ./test_videos/violence --model-path ./checkpoints/fusion_head_best_pr_auc.pth --window-size 8 --device cuda:0
```

The evaluation reports various metrics at the **clip-level**:

- Accuracy
- ROC-AUC
- PR-AUC
- Average precision (AP)

The script also writes the following files in the current working directory:

| Output | Contents |
| --- | --- |
| `preds.npy` | One-dimensional array of violence probabilities, one per evaluated clip. |
| `true_labels.npy` | Matching integer ground-truth labels. |

## Optimal threshold calculation

After generating `preds.npy` and `true_labels.npy`, run:

```bash
python optimal_threshold.py
```

The script searches 500 thresholds between 0 and 1, predicts violence when `probability > threshold`, selects the best **macro F1**, and displays a row-normalized confusion matrix. It prints the selected threshold, score, and matrix.


