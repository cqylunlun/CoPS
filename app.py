import os
from functools import lru_cache

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

import lib
from lib.cops import PromptLearner
from utils.tools import (
    average_neighbor,
    compute_similarity,
    get_fullsize_map,
    get_transform,
    setup_seed,
)


IMAGE_SIZE = 518
FEATURES_LIST = [24]
DEPTH = 8
N_CTX = 12
T_N_CTX = 4
DPAM = 24

CHECKPOINTS = {
    "Trained on VisA": ("visa", "./results/models/visa/epoch_10.pth"),
    "Trained on MVTec": ("mvtec", "./results/models/mvtec/epoch_5.pth"),
}

FIGURES_DIR = "figures"
EXAMPLES = [
    {
        "input_image": "Industrial_Sample_from_MVTec.png",
        "dataset": "MVTec AD",
        "checkpoint": "Trained on VisA",
        "heatmap": "Output_Heatmap_from_MVTec_0.9892697930335999.webp",
    },
    {
        "input_image": "Industrial_Sample_from_VisA.png",
        "dataset": "VisA",
        "checkpoint": "Trained on MVTec",
        "heatmap": "Output_Heatmap_from_VisA_0.9920594096183777.webp",
    },
    {
        "input_image": "Industrial_Sample_from_MPDD.png",
        "dataset": "MPDD",
        "checkpoint": "Trained on VisA",
        "heatmap": "Output_Heatmap_from_MPDD_0.9144860506057739.webp",
    },
    {
        "input_image": "Medical_Sample_from_Br35H.png",
        "dataset": "Br35H",
        "checkpoint": "Trained on VisA",
        "heatmap": "Output_Heatmap_from_Br35H_0.9727131128311157.webp",
    },
    {
        "input_image": "Medical_Sample_from_ColonDB.png",
        "dataset": "ColonDB",
        "checkpoint": "Trained on VisA",
        "heatmap": "Output_Heatmap_from_ColonDB_0.9672988653182983.webp",
    },
]
EXAMPLE_TABLE = [
    [example["input_image"], example["dataset"], example["checkpoint"]]
    for example in EXAMPLES
]

class Args:
    image_size = IMAGE_SIZE


def _figure_path(filename):
    return os.path.join(FIGURES_DIR, filename)


def _score_from_heatmap_name(filename):
    stem, _ = os.path.splitext(filename)
    return float(stem.rsplit("_", 1)[-1])


def _normalize_map(anomaly_map):
    denom = anomaly_map.max() - anomaly_map.min()
    if denom < 1e-8:
        return np.zeros_like(anomaly_map)
    return (anomaly_map - anomaly_map.min()) / denom


def _overlay_heatmap(image, anomaly_map, normalize=False):
    width, height = image.size
    base = np.array(image.convert("RGB"))
    anomaly_map = cv2.resize(anomaly_map, (width, height), interpolation=cv2.INTER_LINEAR)
    if normalize:
        anomaly_map = _normalize_map(anomaly_map)
    heat = (np.clip(anomaly_map, 0.0, 1.0) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(base, 0.5, heat, 0.5, 0)
    return Image.fromarray(blended)


@lru_cache(maxsize=2)
def load_cops(checkpoint_label):
    setup_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset, checkpoint_path = CHECKPOINTS[checkpoint_label]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    hyperparameters = {
        "prompt_length": N_CTX,
        "learnable_text_embedding_depth": DEPTH,
        "learnable_text_embedding_length": T_N_CTX,
        "prt_length": 6,
        "vae_length": 1,
    }
    text_encoder_parameters = hyperparameters if T_N_CTX != 0 else None
    model, _ = lib.load(
        "ViT-L/14@336px",
        device=device,
        design_details=text_encoder_parameters,
    )
    model.visual.DPAM_replace(DPAM_layer=DPAM)
    prompt_learner = PromptLearner(model.cpu(), hyperparameters, FEATURES_LIST)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    prompt_learner.load_state_dict(checkpoint["prompt_learner"], strict=False)
    model = model.to(device).eval()
    prompt_learner = prompt_learner.to(device).eval()

    return device, train_dataset, model, prompt_learner, hyperparameters


@torch.no_grad()
def predict(image, checkpoint_label):
    if image is None:
        raise gr.Error("Please upload an image.")

    device, train_dataset, model, prompt_learner, hyperparameters = load_cops(checkpoint_label)
    preprocess, _ = get_transform(Args())
    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    cls_name = ["object"]

    image_features, patch_features, _, _ = model.encode_image(
        image_tensor,
        FEATURES_LIST,
        DPAM_layer=DPAM,
    )

    z = torch.randn((10,) + image_features.shape, device=device)
    z = z.reshape(-1, z.shape[-1])
    bias = prompt_learner.vae_decoder(z).unsqueeze(1).unsqueeze(1)

    patch_features = [average_neighbor(patch_feature, train_dataset, mode="test") for patch_feature in patch_features]
    fuse_features = torch.stack(patch_features, dim=1).mean(dim=1)

    agg_prototype_n = prompt_learner.extractor(prompt_learner.t_n, fuse_features)
    distribution_n = 1.0 - F.cosine_similarity(fuse_features.unsqueeze(2), agg_prototype_n.unsqueeze(1), dim=-1)
    distance_n, _ = torch.min(distribution_n, dim=2)

    agg_prototype_a = prompt_learner.extractor(prompt_learner.t_a, fuse_features)
    distribution_a = 1.0 - F.cosine_similarity(fuse_features.unsqueeze(2), agg_prototype_a.unsqueeze(1), dim=-1)
    distance_a, _ = torch.min(distribution_a, dim=2)
    agg_prototype = torch.stack([agg_prototype_n, agg_prototype_a], dim=1)

    text_features = prompt_learner(model, image_tensor.shape[0], agg_prototype, bias, cls_name)
    text_features = text_features.mean(dim=2)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    anomaly_map_list = None
    last_patch_sim = None
    for patch_feature in patch_features:
        patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
        patch_sim = compute_similarity(patch_feature, text_features)
        patch_sim = (patch_sim / 0.07).softmax(-1)

        alpha = 0.35 if train_dataset == "visa" else 0.26
        d_n = (distance_n - distance_n.min()) / (distance_n.max() - distance_n.min() + 1e-8)
        d_a = 1.0 - (distance_a - distance_a.min()) / (distance_a.max() - distance_a.min() + 1e-8)
        distance = alpha * d_n + (1.0 - alpha) * d_a
        patch_sim = patch_sim * distance.unsqueeze(-1)

        similarity_map = get_fullsize_map(patch_sim, IMAGE_SIZE)
        anomaly_map = similarity_map[..., 1]
        anomaly_map_list = anomaly_map if anomaly_map_list is None else anomaly_map_list + anomaly_map
        last_patch_sim = patch_sim

    anomaly_map = anomaly_map_list / len(FEATURES_LIST)
    anomaly_map = gaussian_filter(anomaly_map[0].detach().cpu().numpy(), sigma=4)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_sim = compute_similarity(image_features.unsqueeze(1), text_features)
    image_sim = (image_sim / 0.07).softmax(-1)
    image_sim = image_sim[:, 0, 1]

    beta = 1.0 if train_dataset == "visa" else 0.9
    patch_score = torch.amax(last_patch_sim[..., 1], dim=-1)
    score = beta * image_sim + (1.0 - beta) * patch_score
    score_value = float(score.item())

    overlay = _overlay_heatmap(image, anomaly_map, normalize=score_value >= 0.5)
    return score_value, overlay


def load_example(evt: gr.SelectData):
    row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    example = EXAMPLES[row_index]
    input_image = Image.open(_figure_path(example["input_image"])).convert("RGB")
    heatmap = Image.open(_figure_path(example["heatmap"])).convert("RGB")
    score = _score_from_heatmap_name(example["heatmap"])
    return input_image, heatmap, example["checkpoint"], score


with gr.Blocks(
    title="CoPS for Zero-shot Anomaly Detection",
    css="""
    .matched-control {
        min-height: 96px;
    }
    .full-width-button button {
        width: 100%;
    }
    """,
) as demo:
    gr.Markdown("# CoPS for Zero-shot Anomaly Detection")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input image")
        with gr.Column():
            overlay_output = gr.Image(type="pil", label="Anomaly heatmap")
    with gr.Row():
        with gr.Column():
            checkpoint_input = gr.Radio(
                choices=list(CHECKPOINTS.keys()),
                value="Trained on VisA",
                label="Checkpoints",
                elem_classes=["matched-control"],
            )
        with gr.Column():
            score_output = gr.Number(label="Anomaly score", elem_classes=["matched-control"])
    with gr.Row():
        with gr.Column():
            run_button = gr.Button("Run inference", variant="primary", elem_classes=["full-width-button"])
        with gr.Column():
            stop_button = gr.Button("Stop inference", variant="stop", elem_classes=["full-width-button"])
    example_table = gr.Dataframe(
        headers=["Input image", "Dataset name", "Checkpoint"],
        value=EXAMPLE_TABLE,
        datatype=["str", "str", "str"],
        interactive=False,
        wrap=True,
        label="Examples",
    )

    inference_event = run_button.click(
        predict,
        inputs=[image_input, checkpoint_input],
        outputs=[score_output, overlay_output],
    )
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[inference_event])
    example_table.select(
        load_example,
        inputs=None,
        outputs=[image_input, overlay_output, checkpoint_input, score_output],
    )


if __name__ == "__main__":
    demo.launch()
