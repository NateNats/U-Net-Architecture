class EarlyStopping:
    """Hentikan training jika val_loss tidak membaik."""
    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
 
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
        return self.counter >= self.patience