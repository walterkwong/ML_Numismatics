import torch
from ultralytics import YOLO


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    model = YOLO("yolo26n-seg.pt")
    model.to(device)

    print("Model loaded successfully.")
    print("Training model on coin segmentation dataset...")
    model.train(
        data="coin_segmentation_dataset/Coins.v3i.yolo26/data.yaml",
        epochs=100,
        imgsz=640,
    )

    print("Training completed.")
    print("Evaluating model on validation set...")
    metrics = model.val()

    print(metrics.box.maps)

    print("Saving model to 'trained_yoloseg10.pt'...")
    torch.save(model, "trained_yoloseg10.pt")

    print("Exporting model to ONNX...")
    model.export(format="onnx")


if __name__ == "__main__":
    main()