import torch
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification as roberta

def evaluate_model(model: roberta, dataloader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k:v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            pred = torch.argmax(outputs.logits, dim=-1)
            labels = batch['labels']
            correct += (pred == labels).sum().item()
            total += labels.shape[0]
    return correct / total
