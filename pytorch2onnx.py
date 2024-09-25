import torch
from torchvision import models
import argparse
import os

def export_pytorch_to_onnx(pth_path, onnx_path):
    # EfficientNetV2のモデルをロード
    try:
        # 事前学習済みモデルを使用する場合、`pretrained=True`に設定
        model = models.efficientnet_v2_s(pretrained=False)  # 独自の.pthを使用する場合、pretrained=False
    except AttributeError:
        print("torchvisionでEfficientNetV2がサポートされていないバージョンです。最新バージョンにアップデートしてください。")
        return

    # .pthファイルから重みをロード
    try:
        model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        return

    model.eval()

    # ダミー入力の作成
    dummy_input = torch.randn(1, 3, 224, 224)

    # ONNXへのエクスポート
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,  # EfficientNetV2に適したONNXのバージョン
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"ONNX形式へのエクスポートが完了しました: {onnx_path}")
    except Exception as e:
        print(f"ONNXへのエクスポート中にエラーが発生しました: {e}")

def main():
    parser = argparse.ArgumentParser(description="PyTorch (.pth)モデルをONNX形式にエクスポートします。")
    parser.add_argument('--pth_path', type=str, help="PyTorchの.pthファイルのパス")
    parser.add_argument('--onnx_path', type=str, help="出力するONNXファイルのパス")

    args = parser.parse_args()

    if not args.pth_path:
        args.pth_path = input("PyTorchの.pthファイルのパスを入力してください: ").strip()
    if not args.onnx_path:
        default_onnx_path = os.path.splitext(args.pth_path)[0] + ".onnx"
        user_input = input(f"出力するONNXファイルのパスを入力してください（デフォルト: {default_onnx_path}）: ").strip()
        args.onnx_path = user_input if user_input else default_onnx_path

    export_pytorch_to_onnx(args.pth_path, args.onnx_path)

if __name__ == "__main__":
    main()
