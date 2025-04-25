import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
from matplotlib.colors import Normalize


def binarize_output(logit, threshold =0.5):
    if logit.shape[-1] == 1 and logit.ndim == 4:
        logit = logit.permute(0, 3, 1, 2)  # NHWC → NCHW
    binary_pred = (logit > threshold).float()

    return binary_pred

def evaluate(pred, true, eps=1e-6):
    print(pred.dtype, true.dtype)
    pred = pred.int()
    true = true.int()

    intersection = (pred & true).sum(dim=(1, 2))
    union = (pred | true).sum(dim=(1, 2))
    total = torch.numel(pred[0])

    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred.sum(dim=(1,2)) + true.sum(dim=(1,2)) + eps)
    accuracy = (pred == true).float().mean(dim=(1, 2))

    return {
        "Accuracy": accuracy.mean().item(),
        "IoU": iou.mean().item(),
        "Dice": dice.mean().item()
    }

def measure_inference_and_eval(model, x, y, binarize=True, threshold=0.5,
                               warmup=3, repeat=10):
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        # Measure
        elapsed_times = []
        for _ in range(repeat):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            output = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed_times.append(time.time() - start)

    avg_elapsed = sum(elapsed_times) / len(elapsed_times)
    output = output.cpu()

    eval_pred = binarize_output(output, threshold=threshold) if binarize else output

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 念のため評価前にも同期
    metrics = evaluate(eval_pred, y.cpu())

    return avg_elapsed, metrics, output




def summary_metrics_table_binary(inference_times: dict, metrics: dict, title: str = "【評価結果：2値分類】"):
    print(title)
    print("\n【推論時間（ms）】")
    for name, t in inference_times.items():
        print(f"{name:<15}: {t * 1000:.2f} ms")

    print("\n【評価指標（2値分類）】")
    print(f"{'Model':<15} {'Accuracy':>10} {'IoU':>10} {'Dice':>10}")
    print("-" * 50)
    for name, m in metrics.items():
        print(f"{name:<15} {m['Accuracy']:10.4f} {m['IoU']:10.4f} {m['Dice']:10.4f}")


def plot_random_binary_predictions(gt_images, pred_list, labels,
                                         threshold=0.5, num_samples=9,
                                         cmap='viridis', title="GT vs Predictions - Random Samples"):
    """
    2値分類モデルの予測マスクを,GTと並べて可視化（任意のモデル数に対応）

    Parameters:
        gt_images   : ndarray [B, H, W]
        pred_list   : list of ndarray [B, H, W]，各モデルの予測
        labels      : list of str,各モデル名（タイトルに使用）
        threshold   : 2値化閾値
        num_samples : 表示サンプル数
        cmap        : カラーマップ
        title       : 図全体のタイトル
    """
    assert len(pred_list) == len(labels), "pred_listとlabelsの数が一致しません"
    num_models = len(pred_list)
    indices = np.random.choice(len(gt_images), num_samples, replace=False)

    fig, axs = plt.subplots(num_samples, num_models + 1,
                            figsize=(3.5 * (num_models + 1), 3 * num_samples),
                            dpi=120)

    for i, idx in enumerate(indices):
        axs[i, 0].imshow(gt_images[idx], cmap=cmap)
        axs[i, 0].set_title(f"GT #{idx}")
        axs[i, 0].axis('off')

        for j, (pred, label) in enumerate(zip(pred_list, labels)):
            axs[i, j + 1].imshow(pred[idx], cmap=cmap)
            axs[i, j + 1].set_title(f"{label} (th={threshold})")
            axs[i, j + 1].axis('off')

    plt.tight_layout()
    plt.suptitle(title, fontsize=18, y=1.02)
    plt.show()

def plot_prediction_and_error_maps(gt_images, pred_list, labels,
                                         num_samples=1, cmap_pred="viridis", cmap_err="hot",
                                         title="GT vs Predictions and Error Maps"):
    """
    GTと任意数のモデルの出力と誤差マップを上下に並べて可視化（共通カラースケール付き）．

    Parameters:
        gt_images   : ndarray or Tensor [B, H, W]
        pred_list   : list of ndarray or Tensor [B, H, W]
        labels      : list of str, 各モデルのラベル（タイトル用）
        num_samples : int, 表示するランダムサンプル数
        cmap_pred   : str, 予測画像のカラーマップ
        cmap_err    : str, 誤差マップのカラーマップ
        title       : str, 図タイトル
    """
    # NumPy化（Tensor対応）
    if hasattr(gt_images, "cpu"): gt_images = gt_images.cpu().numpy()
    pred_list = [p.cpu().numpy() if hasattr(p, "cpu") else p for p in pred_list]

    # データ長を揃える
    min_len = min([len(gt_images)] + [len(p) for p in pred_list])
    gt_images = gt_images[:min_len]
    pred_list = [p[:min_len] for p in pred_list]

    num_models = len(pred_list)
    indices = np.random.choice(min_len, num_samples, replace=False)

    # 誤差マップ共通スケール取得
    error_max = max([
        np.abs(gt_images[idx] - pred[idx]).max()
        for pred in pred_list
        for idx in indices
    ])

    fig, axs = plt.subplots(num_samples * 2, num_models + 1,
                            figsize=(3.5 * (num_models + 1), 4 * num_samples),
                            dpi=120)
    fig.subplots_adjust(right=0.9)

    norm = Normalize(vmin=0, vmax=error_max)
    sm = plt.cm.ScalarMappable(cmap=cmap_err, norm=norm)
    sm.set_array([])

    for i, idx in enumerate(indices):
        # 上段：GT + 各モデルの予測
        axs[2*i, 0].imshow(gt_images[idx], cmap=cmap_pred)
        axs[2*i, 0].set_title(f"GT #{idx}")
        axs[2*i, 0].axis('off')

        axs[2*i+1, 0].axis('off')  # GTの誤差マップはなし

        for j, (pred, label) in enumerate(zip(pred_list, labels)):
            axs[2*i, j + 1].imshow(pred[idx], cmap=cmap_pred)
            axs[2*i, j + 1].set_title(label)
            axs[2*i, j + 1].axis('off')

            err_map = np.abs(gt_images[idx] - pred[idx])
            axs[2*i+1, j + 1].imshow(err_map, cmap=cmap_err, norm=norm)
            axs[2*i+1, j + 1].set_title(f"Error: {label}")
            axs[2*i+1, j + 1].axis('off')

    # カラーバー
    cax = fig.add_axes([0.92, 0.1, 0.012, 0.8])
    fig.colorbar(sm, cax=cax, label='Absolute Error')

    plt.suptitle(title, fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    plt.show()

def compute_fp_fn_masks(pred, true, threshold=0.5):
    """
    pred, true: (B, 1, H, W) or (B, H, W) tensor
    Returns: FP mask, FN mask (same shape)
    """
    pred = pred.squeeze()
    true = true.squeeze()

    # 連続値なら binarize
    if pred.dtype != torch.bool:
        pred = (pred > threshold).int()

    if true.dtype != torch.bool:
        true = (true > 0.5).int()  # 念のため

    fp = (pred == 1) & (true == 0)
    fn = (pred == 0) & (true == 1)
    return fp, fn

def plot_fp_fn_examples(pred, true, model_name="CNN", num_samples=6):
    fp, fn = compute_fp_fn_masks(pred, true)
    print(f"FP: {fp.sum()}, FN: {fn.sum()}")
    B, H, W = fp.shape

    indices = np.random.choice(B, num_samples, replace=False)

    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))

    for i, idx in enumerate(indices):
        axs[i, 0].imshow(true[idx].squeeze().cpu().numpy(), cmap='gray')
        axs[i, 0].set_title(f"GT #{idx}")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(fp[idx].squeeze().cpu().numpy(), cmap='Reds')
        axs[i, 1].set_title(f"{model_name} FP")
        axs[i, 1].axis('off')

        axs[i, 2].imshow(fn[idx].squeeze().cpu().numpy(), cmap='Blues')
        axs[i, 2].set_title(f"{model_name} FN")
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.suptitle(f"{model_name} False Positives / Negatives", fontsize=18, y=1.02)
    plt.show()