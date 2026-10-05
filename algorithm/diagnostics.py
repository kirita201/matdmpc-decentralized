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

    def __init__(self, log_dir, writer, resume_env_step=None):
        self.writer = writer

        path = Path(log_dir) / "value_diagnostics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        if resume_env_step is not None and path.exists():
            self._discard_future_rows(path, int(resume_env_step))

        exists = path.exists() and path.stat().st_size > 0
        self.file = path.open("a", newline="", encoding="utf-8")
        self.csv = csv.writer(self.file)

        if not exists:
            self.csv.writerow(self.HEADER)
            self.file.flush()

        self.update_index = None
        self.env_step = None
        self.tensorboard_step = None

    @classmethod
    def _discard_future_rows(cls, path, resume_env_step):
        """
        最後のチェックポイントより後の CSV 行を取り除く。

        チェックポイントの step は処理済みなので、
        次に実行するエピソードの step 以上を削除する。
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
                    reader = csv.reader(source_file)

                    for row in reader:
                        if not row:
                            continue

                        if row[0] == "update_index":
                            output.writerow(row)
                        elif int(row[1]) < resume_env_step:
                            output.writerow(row)

            os.replace(temporary_path, path)

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()

    def set_context(
        self,
        update_index,
        env_step,
        tensorboard_step,
    ):
        # CSV では update_index = update() に渡す update_step。
        self.update_index = int(update_index)
        self.env_step = int(env_step)

        # 全 TensorBoard タグで共通の横軸を使用する。
        self.tensorboard_step = int(tensorboard_step)

    @torch.no_grad()
    def values(self, t, **tensors):
        if self.update_index is None:
            raise RuntimeError(
                "update() より先に Diagnostics.set_context() "
                "を呼んでください"
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

    @staticmethod
    def _base_model(model):
        # torch.compile されたモデルでは元の MACLM を参照する。
        return getattr(model, "_orig_mod", model)

    @classmethod
    def _parameter_groups(cls, model):
        base = cls._base_model(model)

        return {
            "q1": (base._Q1,),
            "q2": (base._Q2,),
            "Q_joint": (
                base._mixing1,
                base._mixing2,
            ),
            "reward": (base._reward,),
            "dynamics": (
                base._dynamics,
                base._ln_dyn,
            ),
        }

    @staticmethod
    def _group_norm(tensors):
        """
        複数テンソルを連結した場合と同じ L2 ノルム。
        大きな連結テンソル自体は作らない。
        """
        norms = [
            torch.linalg.vector_norm(
                tensor.detach().float()
            )
            for tensor in tensors
        ]
        return torch.linalg.vector_norm(
            torch.stack(norms)
        ).item()

    @torch.no_grad()
    def gradients(self, model, optimizer):
        if self.tensorboard_step is None:
            raise RuntimeError(
                "update() より先に Diagnostics.set_context() "
                "を呼んでください"
            )

        optimizer_param_ids = {
            id(p)
            for group in optimizer.param_groups
            for p in group["params"]
        }

        for name, modules in self._parameter_groups(model).items():
            grads = [
                p.grad
                for module in modules
                for p in module.parameters()
                if (
                    id(p) in optimizer_param_ids
                    and p.grad is not None
                )
            ]

            # 損失係数がゼロなどで勾配がないグループは
            # 「ゼロだった」と誤解されないよう記録しない。
            if not grads:
                continue

            self.writer.add_scalar(
                f"Grad/{name}",
                self._group_norm(grads),
                self.tensorboard_step,
            )

    @torch.no_grad()
    def weights(self, model, optimizer):
        if self.tensorboard_step is None:
            raise RuntimeError(
                "update() より先に Diagnostics.set_context() "
                "を呼んでください"
            )

        optimizer_param_ids = {
            id(p)
            for group in optimizer.param_groups
            for p in group["params"]
        }

        for name, modules in self._parameter_groups(model).items():
            params = [
                p
                for module in modules
                for p in module.parameters()
                if id(p) in optimizer_param_ids
            ]

            if not params:
                continue

            self.writer.add_scalar(
                f"Weight/{name}",
                self._group_norm(params),
                self.tensorboard_step,
            )

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()