import cv2
import yaml
import torch
import argparse
from time import time
from tqdm import tqdm
from torch.optim import SGD, LBFGS
import numpy as np

from utils.flow_utils import rescale_flow, warp_w_opflow
from config import DATA_PATH


def dot_constraint(x, x0):
    loss = torch.sum(2*x0*x, axis=-1) - 1
    return loss.mean()

def unit_constraint(x):
    loss = (torch.sum(x*x, axis=-1) - 1)**2
    return loss.mean()

def dot_loss(x1, x2):
    loss = torch.sum(x1*x2, axis=-1)
    return loss.mean()

def global_optimization(args):
    eigenvector_path = DATA_PATH / args.dataset / args.eigenvectors
    base_scale = int(args.eigenvectors.split("_")[-1][1:])

    if args.scaling > 1:
        scales = [base_scale, args.scaling]
    else:
        scales = [base_scale]

    splitted_name = args.eigenvectors.split("_")
    global_softmasks = DATA_PATH / args.dataset / f'opt_eig_{args.loss}_{args.loss_c}_e{args.alpha_e}_c{args.alpha_c}' \
                                                  f'_u{args.alpha_u}_l{args.lr}_ni{args.n_iter}_nf1' \
                                                  f'_o{args.optim}_f{args.flow}_s{args.scaling}_' \
                                                  f'eig_{splitted_name[-4]}_{splitted_name[-3]}_{splitted_name[-2]}_{splitted_name[-1]}'
    global_softmasks.mkdir(exist_ok=True)

    (DATA_PATH / args.dataset / f'opt_config_{args.loss}_{args.loss_c}_e{args.alpha_e}_c{args.alpha_c}_'
                                f'u{args.alpha_u}_l{args.lr}_ni{args.n_iter}_nf1_o{args.optim}_f{args.flow}_'
                                f's{args.scaling}_eig_{splitted_name[-3]}_{splitted_name[-2]}_{splitted_name[-1]}.yaml').write_text(yaml.dump(args))

    if args.use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    pbar = tqdm(sorted(eigenvector_path.iterdir()), total=len(list(eigenvector_path.iterdir())))
    for obj_class in pbar:
        global_softmasks_class = global_softmasks / obj_class.name
        global_softmasks_class.mkdir(exist_ok=True)

        eigs_init = {f'eigs_init_{k}': [] for k in scales}
        eigs = {f'eigs_{k}': [] for k in scales}

        star_optim_time = time()
        for scale_id, scale in enumerate(scales):
            fflows = []
            bflows = []

            for eig_idx, eig_file in enumerate(sorted(obj_class.iterdir(), key=lambda path: int(path.name.split('.')[0]))):
                if scale_id == 0:
                    eig = np.load(eig_file)
                else:
                    eig = cv2.resize(eigs[f'eigs_{scales[scale_id-1]}'][eig_idx].reshape(scales[scale_id-1], scales[scale_id-1]).detach().cpu().numpy(), (scale, scale)).flatten()

                if (eig - eig.max()).min() < 0:
                    eig = -(eig - eig.max())
                else:
                    eig = eig - eig.min()

                eig = eig / np.linalg.norm(eig, ord=2)
                eigs_init[f'eigs_init_{scale}'].append(torch.tensor(eig, device=device, dtype=torch.float))
                eigs[f'eigs_{scale}'].append(torch.tensor(eig, device=device, dtype=torch.float, requires_grad=True))

                try:
                    flo = np.load(DATA_PATH / args.dataset / f'flow_1_{args.flow}' / obj_class.name / eig_file.name)
                    flo = np.stack(rescale_flow(flo, scale_factor=scale, renormalize=False)).transpose(1, 2, 0)
                    fflows.append(torch.tensor(flo, dtype=torch.float, device=device))
                except FileNotFoundError:
                    pass

                try:
                    flo = np.load(DATA_PATH / args.dataset / f'flow_reverse_1_{args.flow}' / obj_class.name / eig_file.name)
                    flo = np.stack(rescale_flow(flo, scale_factor=scale, renormalize=False)).transpose(1, 2, 0)
                    bflows.append(torch.tensor(flo, dtype=torch.float, device=device))
                except FileNotFoundError:
                    pass

            eigs_init_pass = torch.stack(eigs_init[f'eigs_init_{scale}'])

            if args.optim == 'sgd':
                optim = SGD(params=eigs[f'eigs_{scale}'], lr=args.lr)
            elif args.optim == 'lbfgs':
                optim = LBFGS(params=eigs[f'eigs_{scale}'], lr=args.lr)

            if args.loss == 'l2':
                loss_fn = torch.nn.MSELoss()
            elif args.loss == 'l1':
                loss_fn = torch.nn.L1Loss()
            elif args.loss == 'huber':
                loss_fn = torch.nn.HuberLoss()
            elif args.loss == 'bce':
                loss_fn = torch.nn.BCEWithLogitsLoss()
            elif args.loss == 'cos':
                loss_fn = dot_loss

            if args.loss_c == 'l2':
                loss_fn_c = torch.nn.MSELoss()
            elif args.loss_c == 'l1':
                loss_fn_c = torch.nn.L1Loss()
            elif args.loss_c == 'huber':
                loss_fn_c = torch.nn.HuberLoss()
            elif args.loss_c == 'bce':
                loss_fn_c = torch.nn.BCEWithLogitsLoss()
            elif args.loss_c == 'cos':
                loss_fn_c = dot_constraint

            for i in range(args.n_iter):
                def closure():
                    eigs_pass = torch.stack(eigs[f'eigs_{scale}'])

                    eigs_fw = []
                    eigs_bw = []

                    for flow, eig in zip(fflows, eigs_pass[:-1]):
                        eigs_fw.append(warp_w_opflow(eig.reshape(scale, scale), flow).flatten())

                    for flow, eig in zip(bflows, eigs_pass[1:]):
                        eigs_bw.append(warp_w_opflow(eig.reshape(scale, scale), flow).flatten())

                    loss = 0

                    loss += args.alpha_e*loss_fn(eigs_pass[1:], torch.stack(eigs_fw))
                    loss += args.alpha_e*loss_fn(eigs_pass[:-1], torch.stack(eigs_bw))

                    loss += args.alpha_c * loss_fn_c(eigs_pass, eigs_init_pass)
                    loss += args.alpha_u * unit_constraint(eigs_pass)
                    return loss.mean()

                closure().backward()
                if args.optim == 'sgd':
                    optim.step()
                elif args.optim == 'lbfgs':
                    optim.step(closure)

                optim.zero_grad()
        end_optim_time = time()
        pbar.set_postfix({'optim_time': f'{(end_optim_time - star_optim_time)/len(eigs[f"eigs_{scales[-1]}"]):.2f}s',
                          'class': obj_class.name})

        rgb_path = DATA_PATH / args.dataset / 'JPEGImages' / obj_class.name
        for eig, im_path in zip(eigs[f'eigs_{scales[-1]}'], sorted(rgb_path.iterdir(), key=lambda path: int(path.name.split('.')[0]))):
            eig = ((eig - eig.min()) / (eig.max() - eig.min())).flatten().detach().cpu().numpy()
            np.save(global_softmasks_class / im_path.with_suffix('.npy').name, eig.reshape(scales[-1], scales[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset for flow estimation", default='davis')
    parser.add_argument('--eigenvectors', help="which features to use",
                        default='pic_eigenvectors_farflow_cityscapes_percentile_90_b0.3_w0.1_s96')
    parser.add_argument('--optim', default='lbfgs', help="which optimizer to use", choices=['sgd', 'lbfgs'])
    parser.add_argument('--flow', help="which optical flow to use", default='arflow_cityscapes_percentile_90')
    parser.add_argument('--loss', default='bce', help="which loss function to use for mask terms",
                        choices=['l1', 'l2', 'huber', 'bce', 'cos'])
    parser.add_argument('--loss-c', default='bce', help="which loss function to use for constraint",
                        choices=['l1', 'l2', 'huber', 'bce', 'cos'])
    parser.add_argument('--alpha-c', type=float, default=1.0, help="initial eigenvector constraints weight")
    parser.add_argument('--alpha-e', type=float, default=0.01, help="eigenvector warping constraints weight")
    parser.add_argument('--alpha-u', type=float, default=0, help="eigenvector unit-vector constraint weight")
    parser.add_argument('--lr', type=float, default=1, help="learning rate")
    parser.add_argument('--n-iter', type=int, default=5, help="number of iterations to use")
    parser.add_argument('--scaling', default=768, type=int)
    parser.add_argument('--use-gpu', action='store_true', help="use gpu for optimization")
    args = parser.parse_args()

    if args.dataset in ['davis', 'segtrackv2']:
        args.alpha_c = 1.0
        args.alpha_e = 0.01
    elif args.dataset == 'fbms59':
        args.alpha_c = 0.1
        args.alpha_e = 0.01

    global_optimization(args)