import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from sklearn.metrics.pairwise import cosine_similarity

from utils.flow_utils import rescale_flow
from config import DATA_PATH


def generate_pic_eigenvectors(args):
    features_path = DATA_PATH / args.dataset / args.features
    scale_factor = np.ceil(int(args.features.split('_')[-1]) / int(args.features.split('_')[-3])).astype(np.int32)

    pic_eigenvectors = DATA_PATH / args.dataset / f'pic_eigenvectors_f{args.flow}_b{args.beta}_w{args.flow_weight}_s{scale_factor}'
    pic_eigenvectors.mkdir(exist_ok=True)

    (DATA_PATH / args.dataset / f'pic_config_f{args.flow}_b{args.beta}_w{args.flow_weight}_s{scale_factor}.yaml').write_text(yaml.dump(args))

    if args.use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    for obj_class in sorted(features_path.iterdir()):
        print(f'Working on class: {obj_class.name}')

        pic_eigenvectors_class = pic_eigenvectors / obj_class.name
        pic_eigenvectors_class.mkdir(exist_ok=True)

        n_features = len(list(obj_class.glob('*.*')))

        pbar = tqdm(sorted(obj_class.iterdir(), key=lambda path: int(path.name.split('.')[0])), total=n_features)
        for feature_file in pbar:
            star_data_time = time()
            features = np.load(feature_file)

            try:
                fflo = np.load(DATA_PATH / args.dataset / f'flow_1_{args.flow}' / obj_class.name / feature_file.name)
                fflo = np.clip(fflo, a_min=-30, a_max=30)
                u, v = rescale_flow(fflo, scale_factor=scale_factor, renormalize=False)
                flows_forward = np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)])
            except FileNotFoundError:
                flows_forward = None

            try:
                bflo = np.load(DATA_PATH / args.dataset / f'flow_reverse_1_{args.flow}' / obj_class.name / feature_file.name)
                bflo = np.clip(bflo, a_min=-30, a_max=30)
                u, v = rescale_flow(bflo, scale_factor=scale_factor, renormalize=False)
                flows_backward = np.hstack([u.reshape(-1, 1), v.reshape(-1, 1)])
            except FileNotFoundError:
                flows_backward = None

            if flows_forward is None:
                flows_forward = flows_backward.copy()
            elif flows_backward is None:
                flows_backward = flows_forward.copy()

            A = cosine_similarity(features, features) \
                + args.flow_weight * cosine_similarity(flows_forward, flows_forward) \
                + args.flow_weight * cosine_similarity(flows_backward, flows_backward)
            A = A/(1 + 2*args.flow_weight)
            A = np.where(A - args.beta > 0, A, 1e-6)

            A = torch.tensor(A, device=device, dtype=torch.float)
            D = torch.diag(torch.sum(A, dim=-1))
            W = torch.linalg.inv(D) @ A

            end_data_time = time()

            start_pic_time = time()
            v_prev = torch.sum(A, dim=-1) / torch.sum(A)
            delta_prev = np.inf
            epsilon = np.inf*torch.ones_like(v_prev)

            while torch.linalg.norm(epsilon, ord=np.inf) > args.stop_thresh:
                prod = torch.mv(W, v_prev)
                v_next = prod / torch.linalg.norm(prod, ord=1)

                delta_next = v_next - v_prev
                epsilon = delta_next - delta_prev

                delta_prev = delta_next
                v_prev = v_next
            end_pic_time = time()

            np.save(pic_eigenvectors_class / feature_file.name, v_next.cpu().numpy())
            pbar.set_postfix({'pic_time': f'{end_pic_time - start_pic_time:.2f}s',
                              'data_time': f'{end_data_time - star_data_time:.2f}s'})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset for flow estimation", default='davis')
    parser.add_argument('--features', help="which features to use", default='dino_features_vit_base_patch_8_size_768')
    parser.add_argument('--flow', help="which flow to use", default='arflow_cityscapes_percentile_90')
    parser.add_argument('--beta', type=float, default=0.3, help="threshold for removing weak edges")
    parser.add_argument('--stop-thresh', type=float, default=1e-6, help="stopping threshold for PIC")
    parser.add_argument('--flow-weight', type=float, default=0.1, help="flow weight")
    parser.add_argument('--use-gpu', action='store_true', help="use flow for mask generation")
    args = parser.parse_args()

    generate_pic_eigenvectors(args)
