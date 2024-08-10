import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

def split_dataset_for_validation(dataset, train_ratio=0.6, val_ratio=0.2):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, len(dataset)))

    return train_dataset, val_dataset, test_dataset



def compute_validation_loss(model, val_dataset, data_collator, tokenizer, batch_size=4):
    model.eval()  # Set model to evaluation mode
    dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)

    loss_fn = CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        inputs = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    model.train()  # Reset model back to training mode
    return avg_loss


def check_for_tensors(obj):
    if isinstance(obj, torch.Tensor):
        print("Found Tensor")
        return True
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if check_for_tensors(value):
                return True
    elif isinstance(obj, list):
        for item in obj:
            if check_for_tensors(item):
                return True
    return False


def convert_tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()  # Convert tensor to list
    elif isinstance(obj, dict):
        return {key: convert_tensor_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensor_to_serializable(item) for item in obj]
    else:
        return obj


