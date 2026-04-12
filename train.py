from datasets.getdata import Dataset
from lib.cops import PromptLearner
from datetime import datetime
from utils.tools import *
from utils.loss import *
from tqdm import tqdm

import os
import lib
import torch
import warnings
import argparse
import numpy as np
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def train(args):
    # Load dataset
    preprocess, target_transform = get_transform(args)
    train_data = Dataset(root=args.train_data_path,
                         transform=preprocess,
                         target_transform=target_transform,
                         dataset_name=args.dataset, )
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # Build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyperparameters = {
        "prompt_length": args.n_ctx,  # 12
        "learnable_text_embedding_depth": args.depth,  # 8
        "learnable_text_embedding_length": args.t_n_ctx,  # 4
        "prt_length": 6,
        "vae_length": 1
    }
    text_encoder_parameters = hyperparameters if args.t_n_ctx != 0 else None
    model_name = "ViT-L/14@336px"  # ViT-B/16@224px ViT-L/14@224px ViT-L/14@336px
    down_ratio = int(model_name.split(os.sep)[1].split('@')[0])
    model, _ = lib.load(model_name, device=device, design_details=text_encoder_parameters)
    args.dpam = None if args.dpam == 0 else args.dpam
    model.visual.DPAM_replace(DPAM_layer=args.dpam)  # v-vv attention in vision encoder
    prompt_learner = PromptLearner(model.cpu(), hyperparameters, args.features_list)  # prompt and text encoder
    model = model.to(device)
    model.eval()
    prompt_learner = prompt_learner.to(device)
    prompt_learner.train()

    # Define loss and optimizer
    loss_bce = F.cross_entropy
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # Start training
    for epoch in tqdm(range(args.epoch)):
        image_loss_list = []
        patch_loss_list = []
        gather_loss_list = []
        scatter_loss_list = []
        kl_loss_list = []
        recon_loss_list = []
        ort_loss_list = []

        pbar = tqdm(train_dataloader)
        for items in pbar:
            image = items['img'].to(device)
            label = items['anomaly']
            cls_name = items['cls_name']
            gt_mask = items['img_mask'].squeeze().to(device)
            gt_mask = torch.where(gt_mask > 0.5, torch.ones_like(gt_mask), torch.zeros_like(gt_mask))  # [8, 518, 518]
            gt_mask_ = torch.nn.functional.max_pool2d(gt_mask.unsqueeze(1).float(), (down_ratio, down_ratio)).squeeze(1)  # [8, 37, 37]
            gt_mask_ = gt_mask_.reshape(image.shape[0], -1)  # [8, 1369]

            # Extract feature from vision encoder
            image_features, patch_features, x_ori, x = model.encode_image(image, args.features_list, DPAM_layer=args.dpam)

            # Fuse global feature in ICTS module
            if hyperparameters['vae_length'] > 0:
                mean, log_var = prompt_learner.vae_encoder(image_features)  # [8, 768]
                std = torch.exp(0.5 * log_var)  # [8, 768]
                z = std * torch.randn((1,) + mean.shape).cuda() + mean  # [8, 768]
                z = z.reshape(-1, z.shape[-1])  # [48, 768]
                bias = prompt_learner.vae_decoder(z).unsqueeze(1).unsqueeze(1)  # [8, 768]
                kl_loss = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1).mean()
                recon_loss = (image_features - bias).pow(2).sum(1).mean()
            else:
                bias = torch.zeros(image_features.shape).cuda()
                kl_loss = torch.tensor(0.0, device=device)
                recon_loss = torch.tensor(0.0, device=device)
            recon_loss = torch.tensor(0.0, device=device) if 'visa' in args.dataset else recon_loss
            kl_loss_list.append(kl_loss.item())
            recon_loss_list.append(recon_loss.item())

            # Inject local features in ESTS module
            patch_features = [average_neighbor(patch_feature, args.dataset, mode='train') for patch_feature in patch_features]
            fuse_features = torch.stack(patch_features, dim=1).mean(dim=1)  # [8, 1369, 768]
            agg_prototype_n = prompt_learner.extractor(prompt_learner.t_n, fuse_features)  # [8, 6, 768]
            distribution_n = 1. - F.cosine_similarity(fuse_features.unsqueeze(2), agg_prototype_n.unsqueeze(1), dim=-1)
            distance_n, _ = torch.min(distribution_n, dim=2)
            agg_prototype_a = prompt_learner.extractor(prompt_learner.t_a, fuse_features)
            distribution_a = 1. - F.cosine_similarity(fuse_features.unsqueeze(2), agg_prototype_a.unsqueeze(1), dim=-1)
            distance_a, _ = torch.min(distribution_a, dim=2)
            agg_prototype = torch.stack([agg_prototype_n, agg_prototype_a], dim=1)  # [8, 2, 6, 768]

            normal_distance_n = distance_n[gt_mask_ == 0]  # [N]
            gather_loss = normal_distance_n.mean()
            gather_loss_list.append(gather_loss.item())
            anomaly_distance_a = distance_a[gt_mask_ != 0]  # [M]
            scatter_loss = anomaly_distance_a.mean() if len(anomaly_distance_a) != 0 else torch.tensor(0.0, device=device)
            scatter_loss_list.append(scatter_loss.item())

            # Extract feature from text encoder
            text_features = prompt_learner(model, image.shape[0], agg_prototype.detach(), bias, cls_name)  # [8, 2, 6, 768]
            ort_loss = torch.tensor(0.0, device=device)
            ort_loss_list.append(ort_loss.item())
            text_features = text_features.mean(dim=2)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute segmentation loss
            similarity_map_list = []
            patch_loss = 0
            for idx, patch_feature in enumerate(patch_features):
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)  # [8, 1370, 768]
                patch_sim = compute_similarity(patch_feature, text_features)  # [8, 1369, 2]
                patch_sim = (patch_sim / 0.07).softmax(-1)
                similarity_map = get_fullsize_map(patch_sim, args.image_size, mode='train').permute(0, 3, 1, 2)  # [8, 2, 518, 518]
                similarity_map_list.append(similarity_map)
            for i in range(len(similarity_map_list)):
                patch_loss += loss_focal(similarity_map_list[i], gt_mask)
                patch_loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt_mask)
                patch_loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1 - gt_mask)
            patch_loss_list.append(patch_loss.item())

            # Compute classification loss
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [8, 768]
            image_sim = compute_similarity(image_features.unsqueeze(1), text_features)
            image_sim = image_sim[:, 0, ...] / 0.07  # [8, 2]
            image_loss = loss_bce(image_sim, label.long().cuda())
            image_loss_list.append(image_loss.item())

            # Update loss and backpropagation
            loss = image_loss + 4 * patch_loss + gather_loss + scatter_loss + kl_loss + recon_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update progress bar
            pbar_str = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - ' \
                       f'epoch [{str(epoch + 1).zfill(2)}/{str(args.epoch).zfill(2)}], ' \
                       f'image_loss:{np.mean(image_loss_list):.4f}, ' \
                       f'patch_loss:{np.mean(patch_loss_list):.4f}, ' \
                       f'gather_loss:{np.mean(gather_loss_list):.4f}, ' \
                       f'scatter_loss:{np.mean(scatter_loss_list):.4f}, ' \
                       f'kl_loss:{np.mean(kl_loss_list):.4f}, ' \
                       f'recon_loss:{np.mean(recon_loss_list):.4f}'
            pbar.set_description_str(pbar_str)

        # Save model
        os.makedirs(args.save_path, exist_ok=True)
        ckpt_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
        parameter_dict = remove_keys_from_model(prompt_learner.state_dict())
        torch.save({"prompt_learner": parameter_dict}, ckpt_path)
        with open('./results/log.txt', 'a', encoding='utf-8') as f:
            f.write(pbar_str + '\n')


if __name__ == '__main__':
    print(f"\033[1;32m{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training start!\033[0m")

    parser = argparse.ArgumentParser("CoPS", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="/root/cqy/dataset/VisA", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./results/models/visa', help='path to save results')
    parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")
    parser.add_argument("--depth", type=int, default=8, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[24], help="features used")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--dpam", type=int, default=24, help="DPAM Layer")
    parser.add_argument("--epoch", type=int, default=10, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")

    args = parser.parse_args()
    setup_seed(0)
    print(args)
    train(args)

    print(f"\033[1;32m{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training finished!\n\n\033[0m")
