import argparse

import cv2
import torch
from torchvision import transforms

from .model import TrafficSignNet


def main():
    parser = argparse.ArgumentParser(description="Classify traffic signs in a video feed")
    parser.add_argument("video", help="Path to video file or camera index")
    parser.add_argument("model", help="Path to trained model")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    model = TrafficSignNet()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    cap = cv2.VideoCapture(int(args.video) if args.video.isdigit() else args.video)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = transforms.functional.to_pil_image(img)
            img = transform(img).unsqueeze(0).to(device)
            output = model(img)
            pred = output.argmax(1).item()
            cv2.putText(frame, f"Pred: {pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Traffic Sign", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
