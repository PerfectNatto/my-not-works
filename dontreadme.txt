# ts_pt_to_executorch.py
import argparse
from pathlib import Path

import torch

def _parse_shape(s: str):
    # "1,3,224,224" -> (1,3,224,224)
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="TorchScript .pt saved by torch.jit.save")
    ap.add_argument("--out", required=True, help="output .pte path")
    ap.add_argument("--input-shape", default="1,3,224,224", help='e.g. "1,3,224,224" (single tensor input)')
    ap.add_argument("--backend", choices=["none", "xnnpack"], default="none")
    ap.add_argument("--strict", action="store_true", help="strict load_state_dict (default: False)")
    args = ap.parse_args()

    pt_path = Path(args.pt)
    out_path = Path(args.out)

    # 1) TorchScript をロード（torch.jit.save された .pt 想定）
    ts = torch.jit.load(str(pt_path), map_location="cpu")
    ts.eval()

    # 2) TorchScript から state_dict を抜く（= 重みを取り出す）
    sd = ts.state_dict()
    if not isinstance(sd, dict) or len(sd) == 0:
        raise RuntimeError("TorchScriptから state_dict を取得できませんでした（空です）。")

    # 3) 元の nn.Module を生成して重みをロード（ここだけ差し替え）
    # ===== ここをあなたの実装に差し替え =====
    # from your_model_file import MyModel
    # model = MyModel(...).eval()
    raise NotImplementedError("ここで元の nn.Module（MyModel）の生成に差し替えてください")
    # =========================================

    missing, unexpected = model.load_state_dict(sd, strict=args.strict)
    if missing:
        print("[warn] missing keys (first 30):", missing[:30])
    if unexpected:
        print("[warn] unexpected keys (first 30):", unexpected[:30])

    # 4) torch.export（ExecuTorchの入口）
    shape = _parse_shape(args.input_shape)
    example_inputs = (torch.randn(*shape),)
    exported = torch.export.export(model, example_inputs)

    # 5) ExecuTorch .pte 書き出し（必要ならXNNPACK lowering）
    if args.backend == "xnnpack":
        from executorch.exir import to_edge_transform_and_lower
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    else:
        import executorch.exir as exir
        edge = exir.to_edge(exported)

    et = edge.to_executorch()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        # write_to_file があればコピーが増えにくい（無ければ buffer を書く）
        if hasattr(et, "write_to_file"):
            et.write_to_file(f)
        else:
            f.write(et.buffer)

    print("wrote:", str(out_path))

if __name__ == "__main__":
    main()
