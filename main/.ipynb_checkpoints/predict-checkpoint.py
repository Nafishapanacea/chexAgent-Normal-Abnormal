import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from dataset import XRayDataset
from transforms import get_val_transform
from model import get_model


def load_model(checkpoint_path, device):
    model = get_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict():

    test_csv = "/home/jupyter-nafisha/X-ray/CSVs/test.csv"
    img_dir = "/home/common"

    # test_csv = '/home/jupyter-nafisha/X-ray/CSVs/NIH_test.csv'
    # img_dir = '/home/jupyter-nafisha/X-ray/Inference_data/NIH-test-dataset'

    # padchest dataset
    # test_csv = '/home/jupyter-nafisha/X-ray/CSVs/PADCHEST_selected_forInf.csv'
    # img_dir = '/home/jupyter-nafisha/X-ray/Inference_data/padchest_selected_dataset'

    # using combined test set
    # test_csv= '/home/jupyter-nafisha/X-ray/CSVs/test_combined.csv'
    # img_dir = '/home/common'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device= "cpu"

    # Load trained model
    checkpoint_path= '/home/jupyter-nafisha/X-ray/checkpoints/best_model_vinBig.pth'
    model = load_model(checkpoint_path, device)

    # Dataset & dataloader
    test_dataset = XRayDataset(test_csv, img_dir, transform=get_val_transform())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Ground-truth labels (Normal=0, Abnormal=1)
    true_labels = test_dataset.df["label"].map({"Normal": 0, "Abnormal": 1}).tolist()
    image_names = test_dataset.df["image_id"].tolist()

    predictions = []

    with torch.no_grad():
        for images, _ in test_loader:  
            images = images.to(device)

            logits = model(images)             # shape: (B, 1)
            probs = torch.sigmoid(logits).squeeze(1)  # shape: (B,)

            preds = (probs >= 0.5).long()      # thresholding for binary prediction

            predictions.extend(preds.cpu().numpy().tolist())

    # Save predictions CSV
    df = pd.DataFrame({
        "image_name": image_names,
        "true_label": true_labels,
        "predicted_label": predictions
    })

    df.to_csv("test_predictions.csv", index=False)
    print("Predictions saved to test_predictions.csv")

    # ---- METRICS ----
    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)

    tn, fp, fn, tp = cm.ravel()
    # Specificity
    specificity = tn / (tn + fp)

    print("\n==== Evaluation Metrics ====")
    print(f"Accuracy     :  {acc:.4f}")
    print(f"Precision    :  {precision:.4f}")
    print(f"Recall       :  {recall:.4f}")
    print(f"Specificity  :  {specificity:.4f}")
    print(f"F1 Score     :  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    predict()
