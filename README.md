# Traffic Sign Neural Network

This project contains a convolutional neural network for classifying German traffic signs (GTSRB) and applying the model to a video feed. The network now includes additional convolutional layers, batch normalization and dropout for improved accuracy.

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
