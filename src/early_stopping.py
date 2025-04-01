import torch

class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.001, save_path="best_vit_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)  # Save best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered. Loading best model.")
                model.load_state_dict(torch.load(self.save_path))  # Load best model
                return True
        return False