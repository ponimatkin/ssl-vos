import cv2
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder

from config import DATA_PATH


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()


def generate_raft_flow(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module.cuda().eval()

    if 'kitti' in args.model.lower():
        model_ds = 'kitti'
    elif 'sintel' in args.model.lower():
        model_ds = 'sintel'
    elif 'chairs' in args.model.lower():
        model_ds = 'chairs'
    elif 'things' in args.model.lower():
        model_ds = 'things'

    rgb_path = DATA_PATH / args.dataset / 'JPEGImages'
    flow_path = DATA_PATH / args.dataset / f'flow_{args.step}_raft_{model_ds}'
    flow_path_reverse = DATA_PATH / args.dataset / f'flow_reverse_{args.step}_raft_{model_ds}'
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
                image1 = load_image((folder / imfile1).as_posix())
                image2 = load_image((folder / imfile2).as_posix())

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                _, flow = model(image1, image2, iters=20, test_mode=True)
                flow = flow[0].permute(1, 2, 0).cpu().numpy()
                np.save(flow_folder / imfile1.with_suffix('.npy').name, flow)

            images = list(reversed(images))
            print(f'Working on folder: {folder.name} in backward direction')
            for imfile1, imfile2 in tqdm(zip(images[:-args.step], images[args.step:]), total=len(images[:-1])):
                image1 = load_image((folder / imfile1).as_posix())
                image2 = load_image((folder / imfile2).as_posix())

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                _, flow = model(image1, image2, iters=20, test_mode=True)
                flow = flow[0].permute(1, 2, 0).cpu().numpy()
                np.save(flow_folder_reverse / imfile1.with_suffix('.npy').name, flow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for flow estimation")
    parser.add_argument('--step', type=int, default=1, help="flow step size")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    generate_raft_flow(args)