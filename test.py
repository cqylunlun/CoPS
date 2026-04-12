from utils.metrics import image_level_metrics, pixel_level_metrics
from scipy.ndimage import gaussian_filter
from datasets.getdata import Dataset
from lib.cops import PromptLearner
from datetime import datetime
from utils.tools import *
from tqdm import tqdm

import os
import lib
import torch
import argparse
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")


@torch.no_grad()
def test(args):
    # Load dataset
    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path,
                        transform=preprocess,
                        target_transform=target_transform,
                        dataset_name=args.dataset, )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    obj_list = test_data.obj_list

    # Result format
    results = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['img_path'] = []
        results[obj]['gt_image'] = []
        results[obj]['pred_image'] = []
        results[obj]['gt_pixel'] = []
        results[obj]['pred_pixel'] = []

    # Build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyperparameters = {
        "prompt_length": args.n_ctx,
        "learnable_text_embedding_depth": args.depth,
        "learnable_text_embedding_length": args.t_n_ctx,
        "prt_length": 6,
        "vae_length": 1
    }
    text_encoder_parameters = hyperparameters if args.t_n_ctx != 0 else None
    model_name = "ViT-L/14@336px"  # ViT-B/16@224px ViT-L/14@224px ViT-L/14@336px
    model, _ = lib.load(model_name, device=device, design_details=text_encoder_parameters)
    args.dpam = None if args.dpam == 0 else args.dpam
    model.visual.DPAM_replace(DPAM_layer=args.dpam)  # v-vv attention in vision encoder
    prompt_learner = PromptLearner(model.cpu(), hyperparameters, args.features_list)  # prompt and text encoder

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"], strict=False)
    model = model.to(device)
    model.eval()
    prompt_learner = prompt_learner.to(device)
    prompt_learner.eval()

    # Start testing
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        gt_mask = items['img_mask']
        gt_mask = torch.where(gt_mask > 0.5, torch.ones_like(gt_mask), torch.zeros_like(gt_mask))

        # Extract feature from vision encoder
        image_features, patch_features, x_ori, x = model.encode_image(image, args.features_list, DPAM_layer=args.dpam)

        # Fuse global feature in ICTS module
        if hyperparameters['vae_length'] > 0:
            z = torch.randn((10,) + image_features.shape).cuda()  # [8, 768]
            z = z.reshape(-1, z.shape[-1])  # [8*10, 768]
            bias = prompt_learner.vae_decoder(z).unsqueeze(1).unsqueeze(1)  # [8, 768]
        else:
            bias = torch.zeros(image_features.shape).cuda()

        # Inject local features in ESTS module
        patch_features = [average_neighbor(patch_feature, args.dataset, mode='test') for patch_feature in patch_features]
        fuse_features = torch.stack(patch_features, dim=1).mean(dim=1)  # [8, 1369, 768]
        agg_prototype_n = prompt_learner.extractor(prompt_learner.t_n, fuse_features)
        distribution_n = 1. - F.cosine_similarity(fuse_features.unsqueeze(2), agg_prototype_n.unsqueeze(1), dim=-1)
        distance_n, _ = torch.min(distribution_n, dim=2)
        agg_prototype_a = prompt_learner.extractor(prompt_learner.t_a, fuse_features)
        distribution_a = 1. - F.cosine_similarity(fuse_features.unsqueeze(2), agg_prototype_a.unsqueeze(1), dim=-1)
        distance_a, _ = torch.min(distribution_a, dim=2)
        agg_prototype = torch.stack([agg_prototype_n, agg_prototype_a], dim=1)  # [8, 2, 6, 768]

        # Extract feature from text encoder
        text_features = prompt_learner(model, image.shape[0], agg_prototype, bias, cls_name)  # [1, 2, 768]
        text_features = text_features.mean(dim=2)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Anomaly segmentation
        for idx, patch_feature in enumerate(patch_features):
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)  # [1, 1370, 768]
            patch_sim = compute_similarity(patch_feature, text_features)  # [1, 1369, 2]
            patch_sim = (patch_sim / 0.07).softmax(-1)

            alpha = 0.26 if 'mvtec' in args.dataset else 0.35
            distance_n = (distance_n - distance_n.min()) / (distance_n.max() - distance_n.min())
            distance_a = 1 - (distance_a - distance_a.min()) / (distance_a.max() - distance_a.min())
            distance = alpha * distance_n + (1 - alpha) * distance_a
            patch_sim = patch_sim * distance.unsqueeze(-1)  # [1, 1369, 2]

            similarity_map = get_fullsize_map(patch_sim, args.image_size)  # [1, 518, 518, 2]
            anomaly_map = similarity_map[..., 1]  # [1, 518, 518]
            if idx == 0:
                anomaly_map_list = anomaly_map
            else:
                anomaly_map_list += anomaly_map
        anomaly_map = anomaly_map_list / len(args.features_list)  # [1, 518, 518]
        anomaly_map = [gaussian_filter(i, sigma=4) for i in anomaly_map.cpu().numpy()]

        # Anomaly detection
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [1, 768]
        image_sim = compute_similarity(image_features.unsqueeze(1), text_features)
        image_sim = (image_sim / 0.07).softmax(-1)
        image_sim = image_sim[:, 0, 1]  # [1,]

        beta = 0.9 if 'mvtec' in args.dataset else 1.0
        patch_sim = torch.amax(patch_sim[..., 1], dim=-1)
        image_sim = beta * image_sim + (1 - beta) * patch_sim

        # Save results
        results[cls_name[0]]['pred_image'].extend(image_sim.cpu().numpy().tolist())
        results[cls_name[0]]['gt_image'].extend(items['anomaly'].numpy().tolist())
        results[cls_name[0]]['pred_pixel'].extend(anomaly_map)
        results[cls_name[0]]['gt_pixel'].extend(gt_mask.numpy().tolist())
        results[cls_name[0]]['img_path'].extend(items['img_path'])

    # Calculate metrics
    result_collect = []
    for num, obj in enumerate(obj_list):
        visualizer(results[obj]['img_path'], results[obj]['pred_pixel'], results[obj]['gt_pixel'], results[obj]['gt_image'], obj)

        image_auroc = image_level_metrics(results, obj, "image-auroc")
        image_ap = image_level_metrics(results, obj, "image-ap")
        pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
        pixel_ap = pixel_level_metrics(results, obj, "pixel-ap")

        result_collect.append({
            "class_name": obj,
            "image_auroc": image_auroc,
            "image_ap": image_ap,
            "pixel_auroc": pixel_auroc,
            "pixel_ap": pixel_ap,
        })

        print(f"{str(num + 1).zfill(2)}", end=" ")
        for key, item in result_collect[-1].items():
            if isinstance(item, str):
                print(f"{key}:{item.ljust(15)}", end="")
            else:
                print(f"{key}:{str(round(item * 100, 2)).ljust(5)} ", end="")
        print("\n", end="")

        mean_metrics = create_csv(result_collect, './results')
        if num == len(obj_list) - 1:
            print("\n")
            print(f">dataset_name:{results[obj]['img_path'][0].split(os.sep)[4].ljust(15)}", end="")
            for key, item in mean_metrics.items():
                print(f"{key[5:]}:{str(round(item * 100, 2)).ljust(5)} ", end="")


if __name__ == '__main__':
    print(f"\033[1;31m{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Test start!\033[0m")

    parser = argparse.ArgumentParser("CoPS", add_help=True)
    parser.add_argument("--data_path", type=str, default="/root/cqy/dataset/MVTec", help="path to test dataset")
    parser.add_argument("--checkpoint_path", type=str, default='./results/models/visa/epoch_10.pth', help='path to checkpoint')
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--depth", type=int, default=8, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[24], help="features used")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--dpam", type=int, default=24, help="DPAM Layer")

    args = parser.parse_args()
    setup_seed(0)
    print(args)
    test(args)

    print(f"\033[1;31m\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Test finished!\n\n\033[0m")