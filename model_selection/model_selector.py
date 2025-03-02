from create_models.custom_model import custom_model
from create_models.easy_model import easy_model
from create_models.custom_CNN import custom_CNN
from create_models.vgg16 import vgg16
from create_models.resnet101 import resnet101
from create_models.custom_ViT import custom_ViT
from tuning_models.tuning_resnet101 import tuning_resnet101
from tuning_models.tuning_ViT import tuning_ViT
from create_models.test_model import test_model
def select_model(model_name, device, num_classes, in_channels):

    # グローバルスコープからインポートされたモデル関数を取得
    all_model_functions = {name: func for name, func in globals().items() if callable(func)}

    # モデル名をリストに分類
    standard_model_names = []
    finetuned_model_names = []

    # チューニングモデルかを判定
    for name in all_model_functions:
        if name.startswith('tuning'):
            finetuned_model_names.append(name)
        else:
            standard_model_names.append(name)

    # モデルがない場合のエラー出力
    if model_name not in standard_model_names + finetuned_model_names:
        raise ValueError(f"Unknown model name: {model_name}")

    model_function = globals()[model_name]

    if model_name in standard_model_names:
        return model_function(device, num_classes, in_channels)
    elif model_name in finetuned_model_names:
        return model_function(device, num_classes)
