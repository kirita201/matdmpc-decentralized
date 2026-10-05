import csv
import os
import tempfile
from pathlib import Path

import torch


class Diagnostics:
    HEADER = [
        "update_index",
        "env_step",
        "horizon_t",
        "metric",
        "mean",
        "std",
        "max_abs",
    ]

    def __init__(self, log_dir, writer, resume_update_step=None):
        self.writer = writer

        path = Path(log_dir) / "value_diagnostics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        if resume_update_step is not None and path.exists():
            self._discard_uncommitted_rows(
                path,
                int(resume_update_step),
            )

        exists = path.exists() and path.stat().st_size > 0
        self.file = path.open("a", newline="", encoding="utf-8")
        self.csv = csv.writer(self.file)

        if not exists:
            self.csv.writerow(self.HEADER)
            self.file.flush()

        self.update_index = None
        self.env_step = None

    @classmethod
    def _discard_uncommitted_rows(cls, path, first_future_update_step):
        """
        チェックポイントより後に書かれた CSV 行を除く。

        異常終了時にはログだけがモデルの保存地点より先に
        進んでいる場合があるため、再実行する更新番号以上の
        行を削除してから追記を再開する。
        """
        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                output = csv.writer(temporary_file)

                with path.open(
                    "r",
                    newline="",
                    encoding="utf-8",
                ) as source_file:
                    for row in csv.reader(source_file):
                        if not row:
                            continue

                        if row[0] == "update_index":
                            output.writerow(row)
                        elif int(row[0]) < first_future_update_step:
                            output.writerow(row)

            os.replace(temporary_path, path)

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

    def set_context(self, update_index, env_step):
        self.update_index = int(update_index)
        self.env_step = int(env_step)

    @torch.no_grad()
    def values(self, t, **tensors):
        if self.update_index is None:
            raise RuntimeError(
                "Diagnostics.set_context() を update() より先に呼んでください"
            )

        for name, tensor in tensors.items():
            if tensor is None:
                continue

            x = tensor.detach().float().reshape(-1)
            if x.numel() == 0:
                continue

            self.csv.writerow([
                self.update_index,
                self.env_step,
                int(t),
                name,
                x.mean().item(),
                x.std(unbiased=False).item(),
                x.abs().max().item(),
            ])

        self.file.flush()

    @torch.no_grad()
    def gradients(self, phase, model, optimizer):
        if self.update_index is None:
            raise RuntimeError(
                "Diagnostics.set_context() を update() より先に呼んでください"
            )

        param_ids = {
            id(p)
            for group in optimizer.param_groups
            for p in group["params"]
        }

        for name, p in model.named_parameters():
            if id(p) not in param_ids or p.grad is None:
                continue

            name = name.removeprefix("_orig_mod.")
            norm = torch.linalg.vector_norm(
                p.grad.detach().float()
            ).item()

            self.writer.add_scalar(
                f"Grad/{phase}/{name}",
                norm,
                self.update_index,
            )

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()