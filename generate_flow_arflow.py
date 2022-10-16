import cv2
import torch
import imageio
import argparse
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from torchvision import transforms
from ARFlow.transforms import sep_transforms

from ARFlow.utils.torch_utils import restore_model
from ARFlow.models.pwclite import PWCLite
from config import DATA_PATH, MAX_FLOW_VAL
from utils.flow_utils import quantize_flow
from ARFlow.utils.warp_utils import flow_warp


def calculate_binary_flow_weights(org_img, warped_img, thr_value, thr_type):

    diff = abs(org_img - warped_img)[0].permute(1, 2, 0).cpu().numpy().mean(2)
    diff = diff / diff.max()
    if thr_type is not None:
        thr = np.percentile(diff, thr_value)
        diff[diff > thr] = 1
        diff[diff <= thr] = 0

    weights = 1 - diff
    return weights


def generate_arflow_flow(args):
    cfg = {
        'model': {
            'upsample': True,
            'n_frames': 2,
            'reduce_dense': True
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
    }

    cfg = EasyDict(cfg)
    model = PWCLite(cfg.model)
    model = restore_model(model, cfg.pretrained_model)
    model = model.eval().cuda()

    input_transform = transforms.Compose([
        sep_transforms.Zoom(*cfg.test_shape),
        sep_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    ])

    if 'kitti' in args.model.lower():
        model_ds = 'kitti'
    elif 'cityscapes' in args.model.lower():
        model_ds = 'cityscapes'

    rgb_path = DATA_PATH / args.dataset / 'JPEGImages'
    flow_path = DATA_PATH / args.dataset / f'flow_{args.step}_arflow_{model_ds}_{args.thr_type}_{args.thr_value}'
    flow_path_reverse = DATA_PATH / args.dataset / f'flow_reverse_{args.step}_arflow_{model_ds}_{args.thr_type}_{args.thr_value}'
    flow_path.mkdir(exist_ok=True)
    flow_path_reverse.mkdir(exist_ok=True)

    with torch.no_grad():
        for folder in sorted(rgb_path.iterdir()):
            images = list(folder.iterdir())
            images = sorted(images, key=lambda path: int(path.name.split('.')[0]))

            flow_folder = flow_path / folder.name
            flow_folder_reverse = flow_path_reverse / folder.name
            flow_folder.mkdir(exist_ok=True)
            flow_folder_reverse.mkdir(exist_ok=True)

            print(f'Working on folder: {folder.name} in forward direction')
            for imfile1, imfile2 in tqdm(zip(images[:-args.step], images[args.step:]), total=len(images[:-args.step])):
                image1 = imageio.imread(imfile1.as_posix()).astype(np.float32)
                image2 = imageio.imread(imfile2.as_posix()).astype(np.float32)

                image1 = input_transform(image1).unsqueeze(0)
                image2 = input_transform(image2).unsqueeze(0)

                img_pair = torch.cat([image1, image2], 1).float().cuda()

                flow = model(img_pair)['flows_fw'][0]
                re_image1 = flow_warp(image2, flow.cpu())
                flow = flow[0].permute(1, 2, 0).cpu().numpy()

                binary_weights = calculate_binary_flow_weights(image1, re_image1, args.thr_value, args.thr_type)
                flow = flow * binary_weights[:,:,None]
                if args.raw:
                    np.save(flow_folder / imfile1.with_suffix('.npy').name, flow)
                else:
                    dx, dy = quantize_flow(flow, max_val=MAX_FLOW_VAL, norm=False)
                    flow = np.stack([dx, dy, np.zeros(dx.shape)], axis=-1)
                    cv2.imwrite((flow_folder / imfile1.name).as_posix(), flow[:, :, [2, 1, 0]])

            images = list(reversed(images))
            print(f'Working on folder: {folder.name} in backward direction')
            for imfile1, imfile2 in tqdm(zip(images[:-args.step], images[args.step:]), total=len(images[:-1])):
                image1 = imageio.imread(imfile1.as_posix()).astype(np.float32)
                image2 = imageio.imread(imfile2.as_posix()).astype(np.float32)

                image1 = input_transform(image1).unsqueeze(0)
                image2 = input_transform(image2).unsqueeze(0)

                img_pair = torch.cat([image1, image2], 1).float().cuda()

                flow = model(img_pair)['flows_fw'][0]
                re_image1 = flow_warp(image2, flow.cpu())
                flow = flow[0].permute(1, 2, 0).cpu().numpy()

                binary_weights = calculate_binary_flow_weights(image1, re_image1, args.thr_value, args.thr_type)
                flow = flow * binary_weights[:,:,None]
                if args.raw:
                    np.save(flow_folder_reverse / imfile1.with_suffix('.npy').name, flow)
                else:
                    dx, dy = quantize_flow(flow, max_val=MAX_FLOW_VAL, norm=False)
                    flow = np.stack([dx, dy, np.zeros(dx.shape)], axis=-1)
                    cv2.imwrite((flow_folder_reverse / imfile1.name).as_posix(), flow[:, :, [2, 1, 0]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='ARFlow/checkpoints/CityScapes/pwclite_ar.tar')
    parser.add_argument('--dataset', help="dataset for flow estimation")
    parser.add_argument('--step', type=int, default=1, help="flow step size")
    parser.add_argument('--test-shape', default=[384, 640], type=int, nargs=2)
    parser.add_argument('--raw', action='store_true', help='generate raw optical flow')
    parser.add_argument('--thr_type', type=str, default='percentile')
    parser.add_argument('--thr_value', type=int, default=90)
    args = parser.parse_args()

    generate_arflow_flow(args)