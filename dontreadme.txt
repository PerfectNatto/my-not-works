# ts_pt_to_pte_draft_export.py
import argparse
from pathlib import Path

import torch


def parse_shape(s: str):
    return tuple(int(p.strip()) for p in s.split(",") if p.strip())


class TorchScriptModuleWrapper(torch.nn.Module):
    def __init__(self, ts_module):
        super().__init__()
        self.ts = ts_module

    def forward(self, x):
        return self.ts(x)


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

    # 2) export 入口用ラップ
    wrapper = TorchScriptModuleWrapper(ts).eval()

    # 3) 例入力（trace時と同じshape）
    x = torch.randn(*parse_shape(args.input_shape), dtype=torch.float32)
    example_inputs = (x,)

    # 4) draft_export（data-dependent を実テンソル併用で通す）
    ep = torch.export.draft_export(wrapper, example_inputs)

    # 必要ならレポート表示（邪魔なら消してOK）
    # print(ep._report)

    # 5) ExecuTorch へ
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
