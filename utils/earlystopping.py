import os
import numpy as np
import torch

class EarlyStopping:
    def __init__(self, dataset_name, timestamp, patience=30, log_dir=None):
        self._filename = f"LAGAT_{timestamp}.pth"
        self._save_dir = os.path.join('checkpoints', dataset_name)
        if log_dir:
            self._save_dir = os.path.join(log_dir, self._save_dir)

        os.makedirs(self._save_dir, exist_ok=True)
        self.save_path = os.path.join(self._save_dir, self._filename)
        print(f"[{self.__class__.__name__}]: Saving model to {self.save_path}")

        self.patience = patience
        self.counter = 0
        self.best_epoch = -1
        self.best_loss = None
        self.best_result = None
        self.early_stop = False

    def step(self, epoch, loss, result, model):
        if self.best_loss is None:
            self.best_epoch = epoch
            self.best_result = result
            self.best_loss = loss
            self.save_checkpoint(model)
        elif result < self.best_result:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience} in epoch {epoch}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if result >= self.best_result:
                self.save_checkpoint(model)
            self.best_epoch = epoch
            self.best_loss = np.min((loss, self.best_loss))
            self.best_result = np.max((result, self.best_result))
            self.counter = 0

        return self.early_stop
    
    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        # print("Model saved to: ", self.save_path)
        torch.save(model.state_dict(), self.save_path)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.save_path, weights_only=True))

