import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from matplotlib.colors import Normalize
import pandas as pd
import seaborn as sns
from scipy.ndimage import sobel
import time 

def compute_high_freq_error(gt, pred, cutoff=0.25):
    """
    高周波誤差（HF_MSE）を評価。
    gt, pred: [B, H, W] or [B, 1, H, W]（float, 同スケール）
    cutoff: 高周波領域の定義（周波数距離の割合）
    Returns: 平均HF_MSE
    """
    if gt.ndim == 4:
        gt = gt.squeeze(1)
        pred = pred.squeeze(1)

    B, H, W = gt.shape
    hf_errors = []

    for i in range(B):
        f_gt = torch.fft.fft2(gt[i])
        f_pred = torch.fft.fft2(pred[i])
        
        f_gt = torch.fft.fftshift(f_gt)
        f_pred = torch.fft.fftshift(f_pred)

        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        center_y, center_x = H // 2, W // 2
        dist = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
        mask = dist > cutoff * max(H, W)

        diff = torch.abs(f_gt - f_pred)
        hf_diff = diff[mask] ** 2
        hf_errors.append(hf_diff.mean().item())

    return np.mean(hf_errors)

def measure_inference_time(model, input, warmup=1, repeat=5):
    with torch.no_grad():
        # warm-up
        for _ in range(warmup):
            _ = model(input)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeat):
            _ = model(input)
        torch.cuda.synchronize()
        return (time.time() - start) / repeat

def evaluate(pred, true):
    mse = np.mean((pred - true) ** 2)
    psnr_val = psnr(true.squeeze(), pred.squeeze(), data_range=1.0)
    ssim_val = ssim(true.squeeze(), pred.squeeze(), data_range=1.0)
    hf_mse = compute_high_freq_error(torch.tensor(true), torch.tensor(pred), cutoff=0.25)
    return  {
        "MSE": mse,
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "HF_MSE": hf_mse
    }

def evaluate_all(pred, true):
    """
    各サンプルごとに MSE, PSNR, SSIM を記録してリストで返す
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.cpu().numpy()
        
    pred = np.squeeze(pred)
    true = np.squeeze(true)

    scores = []
    for p, t in zip(pred, true):
        mse_val = np.mean((p - t) ** 2)
        psnr_val = psnr(t, p, data_range=1.0)
        ssim_val = ssim(t, p, data_range=1.0)
        scores.append({"MSE": mse_val, "PSNR": psnr_val, "SSIM": ssim_val})
    return scores
    
def compute_patchwise_mse(gt, pred, patch_size=4):
    """
    gt, pred: [B, H, W] or [B, 1, H, W]
    Returns: [B, H//p, W//p] の誤差マップ
    """
    if gt.ndim == 4:
        gt = gt.squeeze(1)
        pred = pred.squeeze(1)

    B, H, W = gt.shape
    p = patch_size
    error_map = ((gt - pred) ** 2)

    # パッチ分割 → 平均
    patch_errors = error_map.unfold(1, p, p).unfold(2, p, p)
    patch_errors = patch_errors.contiguous().mean(dim=(-1, -2))  # 平均MSE per patch

    return patch_errors  # shape: [B, H//p, W//p]

def print_inference_and_metrics(inference_times: dict, metrics: dict, title: str = "【評価結果】"):
    print(title)
    print("\n【推論時間（ms）】")
    for name, t in inference_times.items():
        print(f"{name:<15}: {t*1000:.2f} ms")

    print("\n【評価指標】")
    print(f"{'Model':<15} {'MSE':>12} {'PSNR':>10} {'SSIM':>10} {'HF_MSE':>12}")
    print("-" * 65)
    for name, m in metrics.items():
        print(f"{name:<15} {m['MSE']:12.4e} {m['PSNR']:10.2f} {m['SSIM']:10.4f} {m['HF_MSE']:12.4e}")

def boxplot_metric_comparison(metric_dict, metric_name="MSE"):
    """
    モデルごとのスコア分布を箱ひげ図で表示
    metric_dict: {model_name: list of values}
    """
    all_metrics = []
    all_labels = []
    for model, values in metric_dict.items():
        all_metrics.extend(values)
        all_labels.extend([model] * len(values))

    df = pd.DataFrame({metric_name: all_metrics, "Model": all_labels})
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="Model", y=metric_name, data=df)
    plt.title(f"{metric_name} Distribution by Model")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_gt_pred_comparison(gt_images, pred_list, labels, num_samples=9,
                                   cmap="viridis", title="GT vs Predictions"):
    """
    ランダムなサンプルについて、GTと任意数の予測画像を並べて可視化。
    
    Parameters:
    - gt_images: (N, H, W) or (N, 1, H, W)
    - pred_list: list of prediction arrays [(N, H, W), ...]
    - labels: list of strings, model names
    - num_samples: number of rows
    """
    assert len(pred_list) == len(labels), "Number of predictions and labels must match"
    num_models = len(pred_list)
    indices = np.random.choice(len(gt_images), num_samples, replace=False)

    fig, axs = plt.subplots(num_samples, num_models + 1, figsize=(4.2 * (num_models + 1), 2.8 * num_samples))

    for i, idx in enumerate(indices):
        axs[i, 0].imshow(gt_images[idx].squeeze(), cmap=cmap)
        axs[i, 0].set_title(f"GT #{idx}", fontsize=12)
        axs[i, 0].axis('off')

        for j, (pred, label) in enumerate(zip(pred_list, labels)):
            axs[i, j + 1].imshow(pred[idx].squeeze(), cmap=cmap)
            axs[i, j + 1].set_title(label, fontsize=12)
            axs[i, j + 1].axis('off')

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.show()

    

def compute_error(gt, pred, mode="abs"):
    if mode == "abs":
        return np.abs(gt - pred)
    elif mode == "square":
        return (gt - pred) ** 2
    elif mode == "grad":
        grad_gt = sobel(gt)
        grad_pred = sobel(pred)
        return np.abs(grad_gt - grad_pred)
    elif mode == "ssim":
        _, ssim_map = ssim(gt, pred, data_range=1.0, full=True)
        return 1 - ssim_map  # SSIM類似度の逆（誤差として扱う）
    else:
        raise ValueError(f"Unknown error mode: {mode}")
    
def plot_comparison_with_error_maps(gt_images, pred_list, labels,
                                     num_samples=6, cmap_pred="viridis", cmap_err="inferno",
                                     title = None,
                                     error_type="abs"):
    """
    "abs", "square", "grad", "ssim" のいずれかの誤差マップを表示
    """
    assert len(pred_list) == len(labels), "Prediction list and label list must match"
    num_models = len(pred_list)
    indices = np.random.choice(len(gt_images), num_samples, replace=False)

    # 誤差スケール統一のため最大値を取得
    error_max = max([compute_error(gt_images[idx], pred[idx], error_type).max()
                     for pred in pred_list for idx in indices])
    if title is None:
        title = f"GT vs Predictions and Error Maps ({error_type})"

    fig, axs = plt.subplots(num_samples * 2, num_models + 1,
                            figsize=(3 * (num_models + 1), 3.8 * num_samples),
                            dpi=150)

    norm = Normalize(vmin=0, vmax=error_max)
    sm = plt.cm.ScalarMappable(cmap=cmap_err, norm=norm)
    sm.set_array([])

    for i, idx in enumerate(indices):
        gt = gt_images[idx].squeeze()

        axs[2*i, 0].imshow(gt, cmap=cmap_pred)
        axs[2*i, 0].set_title(f"GT #{idx}", fontsize=13)
        axs[2*i, 0].axis('off')

        axs[2*i+1, 0].axis('off')  # GTの誤差マップはなし

        for j, (pred, label) in enumerate(zip(pred_list, labels)):
            p = pred[idx].squeeze()

            axs[2*i, j + 1].imshow(p, cmap=cmap_pred)
            axs[2*i, j + 1].set_title(label, fontsize=13)
            axs[2*i, j + 1].axis('off')

            err_map = compute_error(gt, p, mode=error_type)
            axs[2*i+1, j + 1].imshow(err_map, cmap=cmap_err, norm=norm)
            axs[2*i+1, j + 1].set_title(f"Error: {label}", fontsize=12)
            axs[2*i+1, j + 1].axis('off')

    # カラーバー
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    fig.colorbar(sm, cax=cax, label=f"{error_type} error")

    plt.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 0.91, 0.94])
    plt.show()



def plot_fft_spectrum(image, title="FFT Spectrum", log_scale=True):
    """
    単一画像に対してFFTスペクトルを表示する
    image: [H, W] ndarray
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    if log_scale:
        magnitude = np.log1p(magnitude)  # log(1 + |F|) to enhance visibility

    plt.imshow(magnitude, cmap='magma')
    plt.title(title)
    plt.axis('off')
    plt.colorbar(label='Log Magnitude' if log_scale else 'Magnitude')
    plt.show()


def compare_fft_spectra(gt_images, pred_images_list, labels_list, num_samples=3):
    """
    ランダムなサンプルに対して、GTと各モデル出力のFFTスペクトルを可視化する。
    - gt_images: Ground Truth画像 (B, H, W)
    - pred_images_list: 予測画像リスト（各要素が (B, H, W)）
    - labels_list: 各予測画像に対応するラベルリスト（文字列）
    - num_samples: 可視化するサンプル数

    各行が [GT, pred1, pred2, ..., predN] の構成。
    """
    assert len(pred_images_list) == len(labels_list), "pred_images_listとlabels_listの長さが一致していません。"

    indices = np.random.choice(len(gt_images), num_samples, replace=False)
    num_models = len(pred_images_list) + 1  # +1はGT分

    fig, axs = plt.subplots(num_samples, num_models, figsize=(5 * num_models, 5 * num_samples))

    for i, idx in enumerate(indices):
        images = [gt_images[idx]] + [pred[idx] for pred in pred_images_list]

        for j, img in enumerate(images):
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            mag = np.log1p(np.abs(fshift))
            axs[i, j].imshow(mag, cmap='magma')
            axs[i, j].axis('off')

            if i == 0:
                if j == 0:
                    axs[i, j].set_title("GT", fontsize=12)
                else:
                    axs[i, j].set_title(labels_list[j-1], fontsize=12)

    plt.suptitle("FFT Spectrum Comparison (log scale)", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()
