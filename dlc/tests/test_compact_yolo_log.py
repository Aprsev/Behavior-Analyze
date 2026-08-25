"""Compact YOLO console reporting and terminal sanitization."""

from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest

from dlc.hybrid.compact_yolo import add_compact_training_logger, compact_train
from dlc.hybrid.log_text import sanitize_process_output


class FakeModel:
    def __init__(self):
        self.callbacks = {}
        self.kwargs = None

    def add_callback(self, event, callback):
        self.callbacks[event] = callback

    def train(self, **kwargs):
        self.kwargs = kwargs
        return "result"


class FakeTrainer:
    def __init__(self, epoch):
        self.epoch = epoch
        self.epochs = 12
        self.args = SimpleNamespace(epochs=12)
        self.tloss = [1.0, .5, .25]
        self.metrics = {
            "metrics/precision(B)": .8, "metrics/recall(B)": .7,
            "metrics/mAP50(B)": .75, "metrics/mAP50-95(B)": .5,
        }

    def label_loss_items(self, losses, prefix):
        return dict(zip((f"{prefix}/box_loss", f"{prefix}/cls_loss", f"{prefix}/dfl_loss"), losses))


class CompactYoloLogTests(unittest.TestCase):
    def test_only_each_fifth_and_final_epoch_are_printed(self) -> None:
        model = FakeModel()
        output = io.StringIO()
        with redirect_stdout(output):
            add_compact_training_logger(model, 5)
            callback = model.callbacks["on_fit_epoch_end"]
            for epoch in range(12):
                callback(FakeTrainer(epoch))
        text = output.getvalue()
        self.assertIn("epoch 5/12", text)
        self.assertIn("epoch 10/12", text)
        self.assertIn("epoch 12/12", text)
        self.assertNotIn("epoch 1/12", text)

    def test_verbose_progress_is_forced_off(self) -> None:
        model = FakeModel()
        self.assertEqual(compact_train(model, {"verbose": True}, 5), "result")
        self.assertFalse(model.kwargs["verbose"])

    def test_ansi_and_carriage_returns_are_removed(self) -> None:
        clean = sanitize_process_output("\x1b[Kepoch 1\r\x1b[32mepoch 2\x1b[0m")
        self.assertEqual(clean, "epoch 1\nepoch 2")


if __name__ == "__main__":
    unittest.main()
