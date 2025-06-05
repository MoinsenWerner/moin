# Traffic Sign Neural Network

This project contains a simple convolutional neural network for classifying German traffic signs (GTSRB) and applying the model to a video feed.

## Requirements

Install the dependencies with:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install opencv-python-headless
```

## Training

```bash
python -m traffic_sign.train --data-root ./data --epochs 5 --lr 0.001 --batch-size 64 --device cpu --save-path model.pth
```

The script downloads the GTSRB dataset automatically and saves the model weights to `model.pth`.

## Video Classification

```bash
python -m traffic_sign.predict <video_source> model.pth
```

`<video_source>` can be a path to a video file or a webcam index (e.g., `0`). Press `q` to quit the window.
