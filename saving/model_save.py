import torch
import os

# モデルをONNX形式で保存
def save_model_as_onnx(model, file_path, input_size, device, result_dir, ai_server):
    model.eval()
    dummy_input = torch.randn(1, *input_size).to(device)
    output_path = os.path.join(result_dir, file_path)
    os.makedirs(result_dir, exist_ok=True)
    torch.onnx.export(model, dummy_input, output_path, export_params=True, opset_version=14,
                      do_constant_folding=True, input_names=['input'], output_names=['output'])
    if not ai_server:
        print(f"Model saved to {output_path}")
