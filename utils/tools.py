from lib.transform import image_transform

import cv2
import csv
import os
import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def get_transform(args):
    preprocess = image_transform(args.image_size, is_train=False, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
    preprocess.transforms[0] = transforms.Resize(size=(args.image_size, args.image_size),
                                                 interpolation=transforms.InterpolationMode.BICUBIC,
                                                 max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))

    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])

    return preprocess, target_transform


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualizer(pathes, segmentations, masks_gt, image_gt, cls_name):
    segmentations = np.array(segmentations)
    segmentations = (segmentations - segmentations.min()) / (segmentations.max() - segmentations.min())
    masks_gt = np.array(masks_gt)
    w, h = segmentations.shape[-2:]
    datapath = pathes[0].split(os.sep)[4]

    for idx, path in enumerate(pathes):
        defect = cv2.resize(cv2.imread(path), (w, h))
        target = (masks_gt[idx] * 255).astype('uint8')[0]
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
        segmentations[idx] = (segmentations[idx] - segmentations[idx].min()) / (segmentations[idx].max() - segmentations[idx].min())
        mask = cv2.cvtColor(segmentations[idx], cv2.COLOR_GRAY2BGR)
        mask = (mask * 255).astype('uint8')
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        if cls_name == 'brain':
            target = image_gt[idx] * 255 * np.ones_like(target)

        img_up = np.hstack([defect, target, mask])
        img_up = cv2.resize(img_up, (256 * 3, 256))
        full_path = './results/visual/' + datapath + '/' + cls_name + '/'
        os.makedirs(full_path, exist_ok=True)
        cv2.imwrite(full_path + str(idx + 1).zfill(3) + '.png', img_up)


def compute_similarity(vision_features, text_features):
    B, *_, C = text_features.shape
    text_features = text_features.reshape(B, -1, C).transpose(1, 2)
    similarity = torch.bmm(vision_features, text_features)
    similarity = similarity.reshape(*similarity.shape[:2], 2, -1)
    similarity = similarity.mean(dim=-1)
    return similarity


def get_fullsize_map(similarity, shape, mode='train'):
    side = int(similarity.shape[1] ** 0.5)  # 37
    similarity = similarity.reshape(similarity.shape[0], side, side, -1).permute(0, 3, 1, 2)  # [8, 2, 37, 37]
    if mode == 'train':
        similarity = torch.nn.functional.interpolate(similarity, shape, mode='nearest')  # [8, 2, 518, 518]
    else:
        similarity = torch.nn.functional.interpolate(similarity, shape, mode='bilinear')  # [8, 2, 518, 518]
    similarity = similarity.permute(0, 2, 3, 1)  # [8, 518, 518, 2]
    return similarity


def create_csv(result_collect, run_save_path):
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["class_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    mean_metrics = compute_and_store_final_results(
        run_save_path,
        result_scores,
        result_dataset_names,
        result_metric_names
    )
    return mean_metrics


def compute_and_store_final_results(results_path, results, row_names, column_names):
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics


def average_neighbor(x, dataset_name, mode):
    if mode == 'train':
        K = [3] if 'mvtec' in dataset_name else [5]
    else:
        K = [3] if 'visa' in dataset_name else [5]
    B, N, C = x.shape
    H = W = int(N ** 0.5)
    x = x.transpose(1, 2).reshape(B, C, H, W)  # -> (B, C, H, W)

    outs = []
    for k in K:
        outs.append(F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2))
    out = sum(outs) / len(outs)
    out = out.flatten(2).transpose(1, 2)  # -> (B, N, C)
    return out


def remove_keys_from_model(parameter_dict):
    keys_to_remove = [
        k for k in parameter_dict.keys() if
        k.startswith("projector") or
        k.startswith("extractor_") or
        k.startswith("vae_encoder") or
        k.startswith("token")
    ]
    for k in keys_to_remove:
        parameter_dict.pop(k, None)
    return parameter_dict
