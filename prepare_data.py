import os
import cv2
import glob
import numpy as np
from shutil import copy, copytree, rmtree
from config import DATA_PATH, RAW_DATA_PATH


def prepare_data():

    for dataset in RAW_DATA_PATH.iterdir():
        print(f'Working on {dataset.name}')

        if dataset.name == 'DAVIS':
           ds_path = DATA_PATH / 'davis'
        elif dataset.name == 'SegTrackv2':
           ds_path = DATA_PATH / 'segtrackv2'
        elif dataset.name == 'FBMS59':
            ds_path = DATA_PATH / 'fbms59'
        else:
            continue

        ds_path.mkdir(exist_ok=True)

        rgb_path = ds_path / 'JPEGImages'
        anno_path = ds_path / 'annotations'
        meta_path = ds_path / 'metadata'

        rgb_path.mkdir(exist_ok=True)
        anno_path.mkdir(exist_ok=True)
        meta_path.mkdir(exist_ok=True)

        if dataset.name == 'DAVIS':
            rgb_source = dataset / 'JPEGImages' / '1080p'
            anno_source = dataset / 'Annotations' / '1080p'
            copy(dataset / 'ImageSets' / '1080p' / 'train.txt', meta_path)
            copy(dataset / 'ImageSets' / '1080p' / 'trainval.txt', meta_path)
            copy(dataset / 'ImageSets' / '1080p' / 'val.txt', meta_path)
            copy(dataset / 'Annotations' / 'db_info.yml', meta_path)
        elif dataset.name == 'MoCA_filtered':
            rgb_source = dataset / 'JPEGImages'
            anno_source = dataset / 'Annotations'
        elif dataset.name == 'SegTrackv2':
            rgb_source = dataset / 'JPEGImages'
            anno_source = dataset / 'GroundTruth'
            meta_source = dataset / 'ImageSets'
            for meta_file in meta_source.iterdir():
                copy(meta_file, meta_path)\

            for cls in rgb_source.iterdir():
                for idx, im_name in enumerate(sorted(cls.iterdir())):
                    if 'jpg' in im_name.name:
                        continue

                    im = cv2.imread(im_name.as_posix())
                    cv2.imwrite((cls / f'{idx:05}.jpg').as_posix(), im)
                    im_name.unlink()

            for cls in anno_source.iterdir():
                if cls.name in ['hummingbird', 'drift', 'bmx', 'monkeydog', 'cheetah']:
                    if cls.name == 'cheetah':
                        dir1 = sorted(glob.glob((cls / '1' / '*.bmp').as_posix()))
                        dir2 = sorted(glob.glob((cls / '2' / '*.png').as_posix()))
                    else:
                        dir1 = sorted(glob.glob((cls / '1' / '*.png').as_posix()))
                        dir2 = sorted(glob.glob((cls / '2' / '*.png').as_posix()))

                    for i in range(len(dir1)):
                        im1 = cv2.imread(dir1[i], cv2.IMREAD_GRAYSCALE)
                        im2 = cv2.imread(dir2[i], cv2.IMREAD_GRAYSCALE)
                        ims = np.clip(im1 + im2, 0, 255)
                        cv2.imwrite((cls / f'{i:05}.png').as_posix(), ims)

                    rmtree((cls / '1').as_posix(), ignore_errors=True)
                    rmtree((cls / '2').as_posix(), ignore_errors=True)

                elif cls.name == 'penguin':
                    dir1 = sorted(glob.glob((cls / '1' / '*.png').as_posix()))
                    dir2 = sorted(glob.glob((cls / '2' / '*.png').as_posix()))
                    dir3 = sorted(glob.glob((cls / '3' / '*.png').as_posix()))
                    dir4 = sorted(glob.glob((cls / '4' / '*.png').as_posix()))
                    dir5 = sorted(glob.glob((cls / '5' / '*.png').as_posix()))
                    dir6 = sorted(glob.glob((cls / '6' / '*.png').as_posix()))

                    for i in range(len(dir1)):
                        im1 = cv2.imread(dir1[i], cv2.IMREAD_GRAYSCALE)
                        im2 = cv2.imread(dir2[i], cv2.IMREAD_GRAYSCALE)
                        im3 = cv2.imread(dir3[i], cv2.IMREAD_GRAYSCALE)
                        im4 = cv2.imread(dir4[i], cv2.IMREAD_GRAYSCALE)
                        im5 = cv2.imread(dir5[i], cv2.IMREAD_GRAYSCALE)
                        im6 = cv2.imread(dir6[i], cv2.IMREAD_GRAYSCALE)
                        ims = np.clip(im1 + im2 + im3 + im4 + im5 + im6, 0, 255)
                        cv2.imwrite((cls / f'{i:05}.png').as_posix(), ims)

                    rmtree((cls / '1').as_posix(), ignore_errors=True)
                    rmtree((cls / '2').as_posix(), ignore_errors=True)
                    rmtree((cls / '3').as_posix(), ignore_errors=True)
                    rmtree((cls / '4').as_posix(), ignore_errors=True)
                    rmtree((cls / '5').as_posix(), ignore_errors=True)
                    rmtree((cls / '6').as_posix(), ignore_errors=True)

                if cls.name == 'worm':
                    for anno_name in sorted(cls.iterdir()):
                        if anno_name.name == '0000000000.png':
                            anno_name.unlink()

                for idx, anno_name in enumerate(sorted(cls.iterdir())):
                    if anno_name.is_dir():
                        continue

                    os.rename(anno_name, cls / f'{idx:05}.png')



        elif dataset.name == 'FBMS59':
            train_path = dataset / 'Trainingset'
            test_path = dataset / 'Testset'

            for path_iter in [train_path, test_path]:
                for cls in path_iter.iterdir():
                    rgb_path_cls = rgb_path / cls.name
                    anno_path_cls = anno_path / cls.name
                    rgb_path_cls.mkdir(exist_ok=True)
                    anno_path_cls.mkdir(exist_ok=True)

                    for im in sorted(cls.iterdir()):
                        if im.is_dir():
                            is_ppm = len(list(im.glob('*.ppm'))) > 0

                            for mask in sorted(im.iterdir()):
                                if '.dat' in mask.suffix:
                                    continue

                                if is_ppm and mask.suffix == '.pgm':
                                    continue

                                if 'PROB' in mask.name:
                                    continue

                                if is_ppm:
                                    mask_im = cv2.imread(mask.as_posix())
                                    pmask = cv2.cvtColor(mask_im, cv2.COLOR_RGB2GRAY)
                                    final_mask = ((~(pmask/255 > 0.9))*255).astype(np.uint8)
                                    cv2.imwrite((anno_path_cls / f'{int(mask.name.split(".")[0].split("_")[1]) - 1:05}.png').as_posix(), final_mask)
                                else:
                                    mask_im = cv2.imread(mask.as_posix())
                                    pmask = cv2.cvtColor(mask_im, cv2.COLOR_RGB2GRAY)
                                    if cls.name == 'marple2':
                                        final_mask = ((pmask/255 > 0.4) * 255).astype(np.uint8)
                                    elif cls.name == 'marple7':
                                        final_mask = ((pmask/255 > 0.05) * 255).astype(np.uint8)
                                    else:
                                        final_mask = (pmask.astype(bool)*255).astype(np.uint8)
                                    if cls.name == 'tennis':
                                        mdx = ''.join([s for s in mask.name if s.isdigit()])
                                        mdx = int(mdx) - 454
                                        cv2.imwrite((anno_path_cls / f'{mdx:05}.png').as_posix(), final_mask)
                                    elif cls.name == 'marple4':
                                        cv2.imwrite((anno_path_cls / f'{int(mask.name.split(".")[0].split("_")[1]) - 324:05}.png').as_posix(), final_mask)
                                    else:
                                        cv2.imwrite((anno_path_cls / f'{int(mask.name.split(".")[0].split("_")[1]) - 1:05}.png').as_posix(), final_mask)
                        else:
                            if im.suffix == '.bmf':
                                continue

                            if cls.name == 'tennis':
                                idx = ''.join([s for s in im.name if s.isdigit()])
                                idx = int(idx) - 454
                                os.rename(im, cls / f'{idx:05}.jpg')
                                copy(cls / f'{idx:05}.jpg', rgb_path_cls / f'{idx:05}.jpg')
                            elif cls.name == 'marple4':
                                os.rename(im, cls / f'{int(im.name.split(".")[0].split("_")[-1]) - 324:05}.jpg')
                                copy(cls / f'{int(im.name.split(".")[0].split("_")[-1]) - 324:05}.jpg', rgb_path_cls / f'{int(im.name.split(".")[0].split("_")[-1]) - 324:05}.jpg')
                            else:
                                os.rename(im, cls / f'{int(im.name.split(".")[0].split("_")[-1]) - 1:05}.jpg')
                                copy(cls / f'{int(im.name.split(".")[0].split("_")[-1]) - 1:05}.jpg', rgb_path_cls / f'{int(im.name.split(".")[0].split("_")[-1]) - 1:05}.jpg')


        if dataset.name != 'FBMS59':
            for obj_rgb in rgb_source.iterdir():
                copytree(obj_rgb, rgb_path / obj_rgb.name)

            for obj_anno in anno_source.iterdir():
                if dataset.name == 'MoCA_filtered':
                    copy(obj_anno, anno_path / obj_anno.name)
                else:
                    copytree(obj_anno, anno_path / obj_anno.name)

if __name__ == "__main__":
    prepare_data()
