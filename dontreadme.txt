# demucs_ts_pt_to_executorch.py
import argparse
from pathlib import Path
import torch

def parse_shape(s: str):
    # "1,2,44100" -> (1,2,44100)
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())

class TSWrapper(torch.nn.Module):
    """torch.export が ScriptModule を直で嫌う場合の回避用（それでも落ちることはあります）"""
    def __init__(self, ts):
        super().__init__()
        self.ts = ts

    def forward(self, x):
        return self.ts(x)

def do_torch_export(model, example_inputs, strict: bool):
    # PyTorchのバージョン差分に備えて strict 引数あり/なし両対応
    try:
        return torch.export.export(model, example_inputs, strict=strict)
    except TypeError:
        return torch.export.export(model, example_inputs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="TorchScript .pt (saved by torch.jit.save)")
    ap.add_argument("--out", required=True, help="output .pte path")
    ap.add_argument("--input-shape", default="1,2,44100", help='e.g. "1,2,44100"')
    ap.add_argument("--backend", choices=["none", "xnnpack"], default="none")
    ap.add_argument("--strict-export", action="store_true", help="use strict=True in torch.export when supported")
    args = ap.parse_args()

    pt_path = Path(args.pt)
    out_path = Path(args.out)

    # 1) TorchScript をロード
    ts = torch.jit.load(str(pt_path), map_location="cpu")
    ts.eval()

    # 2) ダミー入力（Demucs想定で (B, C, T) をデフォルトにしています）
    shape = parse_shape(args.input_shape)
    x = torch.randn(*shape, dtype=torch.float32)
    example_inputs = (x,)

    # 3) torch.export（まず直で試して、ダメなら wrapper を試す）
    try:
        ep = do_torch_export(ts, example_inputs, strict=args.strict_export)
    except Exception as e1:
        wrapped = TSWrapper(ts).eval()
        try:
            ep = do_torch_export(wrapped, example_inputs, strict=args.strict_export)
        except Exception as e2:
            raise RuntimeError(
                "TorchScript(.pt) -> torch.export が失敗しました。\n"
                "これはよくあるパターンで、ExecuTorch の標準入口が eager nn.Module を前提にしているためです。\n"
                f"[direct error]\n{e1}\n\n[wrapper error]\n{e2}\n"
            ) from e2

    # 4) ExecuTorch へ（必要なら XNNPACK lowering）
    if args.backend == "xnnpack":
        from executorch.exir import to_edge_transform_and_lower
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    else:
        import executorch.exir as exir
        edge = exir.to_edge(ep)

    et = edge.to_executorch()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        if hasattr(et, "write_to_file"):
            et.write_to_file(f)
        else:
            f.write(et.buffer)

    print("wrote:", str(out_path))

if __name__ == "__main__":
    main()
