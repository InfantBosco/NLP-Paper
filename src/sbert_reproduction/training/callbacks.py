class Callback:
    def on_epoch_end(self, epoch: int, metrics: dict):
        pass

class LoggingCallback(Callback):
    def on_epoch_end(self, epoch: int, metrics: dict):
        print(f"[Epoch {epoch}] Metrics: {metrics}")
