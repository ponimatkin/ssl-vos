import cv2
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

from config import DATA_PATH
from utils.mask_utils import BilateralSolver, BilateralGrid

def crf_process_class(obj_class, method_path, method_path_masks, img_path, args):
    method_path_class = method_path / obj_class.name
    method_path_masks_class = method_path_masks / obj_class.name
    img_path_class = img_path / obj_class.name
    method_path_masks_class.mkdir(exist_ok=True)

    n_frames = len(list(method_path_class.iterdir()))

    masks = []
    imgs = []
    for frame_id in range(n_frames):
        masks.append(np.load((method_path_class / f'{frame_id:05}.npy').as_posix()).reshape(resolution, resolution))
        imgs.append(cv2.imread((img_path_class / f'{frame_id:05}.jpg').as_posix())[..., ::-1])

    for i in range(len(masks)):
        eig = masks[i]
        img = imgs[i]

        if args.dataset == 'davis':
            img = cv2.resize(img, (854, 480))

        clustering = KMeans(n_clusters=2, n_init=50).fit(eig.reshape(-1, 1))
        assigned_labels = clustering.labels_

        if np.sum(assigned_labels == 0) < np.sum(assigned_labels == 1):
            obj_id = 0
        else:
            obj_id = 1

        mask = (255 * (assigned_labels.reshape(resolution, resolution) == obj_id)).astype(np.float32)
        if args.dataset == 'davis':
            mask = cv2.resize(mask, (854, 480), interpolation=cv2.INTER_LINEAR)
        else:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0).astype(np.float32)

        grid_params = {
            'sigma_luma': args.sigma_l,  # Brightness bandwidth
            'sigma_chroma': args.sigma_c,  # Color bandwidth
            'sigma_spatial': args.sigma_s  # Spatial bandwidth
        }

        bs_params = {
            'lam': args.lam,  # The strength of the smoothness parameter
            'A_diag_min': 1e-5,  # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
            'cg_tol': 1e-5,  # The tolerance on the convergence in PCG
            'cg_maxiter': 25  # The number of PCG iterations
        }

        grid = BilateralGrid(img, **grid_params)
        t = mask.reshape(-1, 1).astype(np.double)
        c = (0.999*np.ones((mask.shape))).reshape(-1, 1).astype(np.double)
        output_solver = BilateralSolver(grid, bs_params).solve(t, c)
        mask = output_solver.reshape(mask.shape) > args.solver_t
        cv2.imwrite((method_path_masks_class / f'{i:05}.png').as_posix(), (255*mask).astype(np.uint8))

def crf_postprocess(args):
    method_path = DATA_PATH / args.dataset / args.method
    img_path = DATA_PATH / args.dataset / 'JPEGImages'

    method_path_masks = DATA_PATH / args.dataset / (args.method.replace('opt_eig', 'opt_masks'))
    method_path_masks.mkdir(exist_ok=True)

    Parallel(n_jobs=args.n_jobs)(delayed(crf_process_class)(obj_class, method_path, method_path_masks, img_path, args) for obj_class in
                                 tqdm(sorted(method_path.iterdir()), total=len(list(method_path.iterdir()))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="which dataset to use")
    parser.add_argument('--method', help="which method eigenmasks to use")
    parser.add_argument('--n-jobs', type=int, default=40, help="number of jobs")
    parser.add_argument('--sigma-l', type=float, default=4.0, help="sigma luma")
    parser.add_argument('--sigma-c', type=float, default=4.0, help="sigma chrome")
    parser.add_argument('--sigma-s', type=float, default=10.0, help="sigma spatial")
    parser.add_argument('--lam', type=float, default=256, help="strength of the smoothness parameter")
    parser.add_argument('--solver_t', type=float, default=0.375, help="solver threshold")
    args = parser.parse_args()

    if args.dataset in ['davis', 'segtrackv2', 'fbms59']:
        resolution = 768 if args.dataset == 'davis' else 480
        args.resolution = resolution
        args.sigma_l = 4.0
        args.sigma_c = 4.0
        args.sigma_s = 10.0
        args.lam = 256
        args.solver_t = 0.375

    crf_postprocess(args)