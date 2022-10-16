import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import transforms as pth_transforms

import vision_transformer as vits
from config import DATA_PATH


def get_features(model, x):
    with torch.no_grad():
        x = model.prepare_tokens(x)
        for i, blk in enumerate(model.blocks):
            if i < len(model.blocks) - 1:
                x = blk(x)
        last_blk = model.blocks[-1]

        x = last_blk.norm1(x)
        B, N, C = x.shape
        qkv = last_blk.attn.qkv(x).reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        return q, k, v

def generate_dino_masks(args):
    # TODO add custom checkpoint loading
    model = vits.__dict__[args.arch](
        patch_size=args.patch_size, num_classes=0
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to('cuda')

    path = None
    if args.arch == "vit_small" and args.patch_size == 16:
        path = "dino_deitsmall16_pretrain.pth"
    elif args.arch == "vit_small" and args.patch_size == 8:
        path = "dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
    elif args.arch == "vit_base" and args.patch_size == 16:
        path = "dino_vitbase16_pretrain.pth"
    elif args.arch == "vit_base" and args.patch_size == 8:
        path = "dino_vitbase8_pretrain.pth"
    if path is not None:
        state_dict = torch.load(f'models/{path}')
        model.load_state_dict(state_dict, strict=True)

    transform = pth_transforms.Compose(
        [
            pth_transforms.ToPILImage(),
            pth_transforms.Resize((args.resize, args.resize), interpolation=3),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )

    rgb_path = DATA_PATH / args.dataset / 'JPEGImages'
    dino_features = DATA_PATH / args.dataset / f'dino_features_{args.arch}_patch_{args.patch_size}_size_{args.resize}'
    dino_features.mkdir(exist_ok=True)

    for obj_class in sorted(rgb_path.iterdir()):
        print(f'Working on class: {obj_class.name}')

        dino_features_class = dino_features / obj_class.name
        dino_features_class.mkdir(exist_ok=True)

        n_rgb_frames = len(list(obj_class.glob('*.*')))

        for im_file in tqdm(sorted(obj_class.iterdir(), key=lambda path: int(path.name.split('.')[0])), total=n_rgb_frames):
            im = cv2.imread(im_file.as_posix())[:, :, [2, 1, 0]]
            img = transform(im).unsqueeze(0).cuda()

            _, out, _ = get_features(model, img)
            out = out[0].cpu().detach().numpy()
            out = out[:, 1:, :]
            out = np.concatenate(out, axis=-1)
            np.save(dino_features_class / im_file.with_suffix('.npy').name, out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset for flow estimation")
    parser.add_argument('--arch', help="DINO model to use", default='vit_base')
    parser.add_argument("--patch-size", default=8, type=int, help="Patch resolution of the model")
    parser.add_argument("--resize", default=768, type=int, help="input image size")
    args = parser.parse_args()

    assert args.resize % args.patch_size == 0

    generate_dino_masks(args)
