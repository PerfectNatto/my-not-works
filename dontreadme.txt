# ts_pt_to_pte.py
import argparse
from pathlib import Path

import torch


def parse_shape(s: str):
    # "1,2,44100" -> (1,2,44100)
    parts = [p.strip() for p in s.split(",")]
    return tuple(int(p) for p in parts if p)


class TorchScriptModuleWrapper(torch.nn.Module):
    """
    TorchScript(ScriptModule) を torch.export の入口として扱うためのラッパーです。
    forward は ts(*args) をそのまま呼びます。
    """
    def __init__(self, ts_module):
        super().__init__()
        self.ts = ts_module

    def forward(self, x):
        return self.ts(x)


def torch_export(module: torch.nn.Module, example_inputs: tuple):
    # PyTorch のバージョン差分対策（strict 引数が無い場合がある）
    try:
        return torch.export.export(module, example_inputs, strict=False)
    except TypeError:
        return torch.export.export(module, example_inputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="TorchScript .pt saved by torch.jit.save")
    ap.add_argument("--out", required=True, help="output .pte path")
    ap.add_argument("--input-shape", default="1,2,44100", help='e.g. "1,2,44100" (B,C,T)')
    args = ap.parse_args()

    pt_path = Path(args.pt)
    out_path = Path(args.out)

    # 1) TorchScript をロード
    ts = torch.jit.load(str(pt_path), map_location="cpu")
    ts.eval()

    # 2) torch.export できる形にラップ
    wrapper = TorchScriptModuleWrapper(ts).eval()

    # 3) 例入力（trace時と同じshapeにすること）
    x = torch.randn(*parse_shape(args.input_shape), dtype=torch.float32)
    example_inputs = (x,)

    # 4) torch.export -> Edge -> ExecuTorch
    exported = torch_export(wrapper, example_inputs)

    import executorch.exir as exir
    edge = exir.to_edge(exported)
    et = edge.to_executorch()

    # 5) .pte 書き出し
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        if hasattr(et, "write_to_file"):
            et.write_to_file(f)
        else:
            f.write(et.buffer)

    print("wrote:", str(out_path))


if __name__ == "__main__":
    main()
