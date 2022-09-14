print('\n Loading...')
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import math
import re
import time
from importlib import import_module

import cv2
import numpy as np
import torch
from torch.nn import functional as F

DEVICE, MODEL, GPU_FP16 = [None] * 3
MAXC, THR = 7, 0.02


def LoadModel(modelDir: str, _gpu_fp16: bool = False):
    global DEVICE, MODEL, GPU_FP16
    # run in cmd:  set CUDA_VISIBLE_DEVICES=""  to force cpu mode
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        GPU_FP16 = _gpu_fp16
        if GPU_FP16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    if not modelDir:
        available = sorted([f for f in os.listdir() if re.match(r'^rife\d{2,3}$', f)])
        modelDir = available[-1] if available else 'train_log'
    try:
        Model = import_module(modelDir + '.RIFE_HDv3').Model
        MODEL = Model()
        MODEL.load_model(modelDir, -1)
    except Exception as e:
        print('ERROR while loading model:', e)
        raise
    if not hasattr(MODEL, 'version'):
        MODEL.version = 0
    MODEL.eval()
    MODEL.device()
    return f"RIFE v{MODEL.version or '??'}", str(DEVICE).upper()


def StartModel():
    t = torch.zeros(1, 3, 64, 64).to(DEVICE)
    MODEL.inference(t, t)


def RatioSplit(img0: torch.Tensor, img1: torch.Tensor, ratio: float | list, scale: float = 1.0):
    if MODEL.version >= 3.9:
        if type(ratio) == list:
            return [MODEL.inference(img0, img1, r, scale) for r in ratio]
        else:
            return MODEL.inference(img0, img1, ratio, scale)
    else:
        if type(ratio) == list:
            ratios = ratio
            mem_size = 2 if len(ratio) < 6 else min(4, math.sqrt(len(ratio)))
        else:
            ratios = [ratio]
            mem_size = 0
        done = {}
        results = []
        for r in ratios:
            img0_r = 0.0
            img1_r = 1.0
            tmp_img0 = img0
            tmp_img1 = img1
            for inference_cycle in range(MAXC):
                mid_r = (img0_r + img1_r) / 2
                if mid_r not in done:
                    img = MODEL.inference(tmp_img0, tmp_img1, scale)
                    if inference_cycle < mem_size:
                        done[mid_r] = img  # saving for possible later reuse
                else:
                    img = done[mid_r]  # reusing
                if r - (THR / 2) <= mid_r <= r + (THR / 2):
                    break
                if r > mid_r:
                    tmp_img0 = img
                    img0_r = mid_r
                else:
                    tmp_img1 = img
                    img1_r = mid_r
            results.append(img)
        done.clear()
        if type(ratio) == list:
            return results
        else:
            return results[0]


def MultiplyFPS(img0: torch.Tensor, img1: torch.Tensor, x: int, scale: float = 1.0) -> list:
    if MODEL.version >= 3.9:
        return [MODEL.inference(img0, img1, (i + 1) / x, scale) for i in range(x - 1)]
    else:
        if x in (2, 4, 8, 16, 32):
            img_list = [img0, img1]
            for _ in range(int(math.log2(x))):
                tmp = []
                for i in range(len(img_list) - 1):
                    mid = MODEL.inference(img_list[i], img_list[i + 1], scale)
                    tmp.append(img_list[i])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp
            return img_list[1:-1]
        else:
            ratios = [(i + 1) / x for i in range(x - 1)]
            return RatioSplit(img0, img1, ratios, scale)


def ReadImage(dir: str, name: str, to_torch: bool = True):
    if name.lower().endswith('.exr'):
        img = cv2.imread(os.path.join(dir, name), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    else:
        img = cv2.imread(os.path.join(dir, name), cv2.IMREAD_UNCHANGED)
    h, w, _ = img.shape
    is_uint16 = True if img.dtype == np.uint16 else False

    if to_torch:
        if img.dtype == np.uint8:
            img = (torch.tensor(img.transpose(2, 0, 1)).to(DEVICE) / 255).unsqueeze(0)
        elif img.dtype == np.uint16:
            img = (torch.tensor(img.transpose(2, 0, 1).astype(np.float32)).to(DEVICE) / (2**16 - 1)).unsqueeze(0)
        else:
            img = (torch.tensor(img.transpose(2, 0, 1)).to(DEVICE)).unsqueeze(0)
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img = F.pad(img, padding).half() if GPU_FP16 else F.pad(img, padding)
    return img, h, w, is_uint16


def WriteImage(dir: str, name: str, img: torch.Tensor, h: int, w: int, is_uint16: bool, _write_kwargs: dict = {}):
    write_kwargs = {'exr': 'B44A', 'png': 1, 'jpg_q': 98, 'jpg_s': '444'}
    write_kwargs.update(_write_kwargs)
    img = img.to(torch.float32)
    if not os.path.exists(dir):
        os.mkdir(dir)
    path = os.path.join(dir, name)
    if name.lower().endswith('.exr'):
        exec(f"write_kwargs['exr'] = cv2.IMWRITE_EXR_COMPRESSION_{write_kwargs['exr']}")
        cv2.imwrite(path, img[0].cpu().numpy().transpose(1, 2, 0)[:h, :w],
                    [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
                    cv2.IMWRITE_EXR_COMPRESSION, write_kwargs['exr']])
    elif name.lower().endswith(('.png', '.tif', '.tiff')) and is_uint16:
        cv2.imwrite(path, (img[0] * (2**16 - 1)).cpu().numpy().astype(np.uint16).transpose(1, 2, 0)[:h, :w],
                    [cv2.IMWRITE_PNG_COMPRESSION, write_kwargs['png'],
                     cv2.IMWRITE_TIFF_COMPRESSION, 1])  # no tiff compression for faster saving
    elif name.lower().endswith(('.jpg', '.jpeg')):
        exec(f"write_kwargs['jpg_s'] = cv2.IMWRITE_JPEG_SAMPLING_FACTOR_{write_kwargs['jpg_s']}")
        cv2.imwrite(path, (img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w],
                    [cv2.IMWRITE_JPEG_QUALITY, write_kwargs['jpg_q'],
                     cv2.IMWRITE_JPEG_OPTIMIZE, 0,  # 0 is good, 1 is 30% longer with only 5% smaller size
                     cv2.IMWRITE_JPEG_SAMPLING_FACTOR, write_kwargs['jpg_s']])
    else:
        cv2.imwrite(path, (img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w],
                    [cv2.IMWRITE_PNG_COMPRESSION, write_kwargs['png'],
                     cv2.IMWRITE_TIFF_COMPRESSION, 1])  # no tiff compression for faster saving


def main():
    global MAXC, THR
    import argparse
    p = argparse.ArgumentParser(description='AI interpolation for a pair of images using RIFE model')
    p.add_argument('-img', nargs=2, type=str, required=True, help='paths to img0 and img1: the 2 input images')
    p.add_argument('-x', default=2, type=int, choices=range(2, 33), metavar='[2-32]',
                   help='FPS multiplier, will render x-1 new images, unused when -r given, default 2')
    p.add_argument('-r', type=float, metavar='(0-1)', help='interpolation ratio between two images, default 0.5')
    p.add_argument('-model', type=str, help='folder with RIFE model files, default is the latest rife** or train_log')
    p.add_argument('-scale', default=1.0, type=float, choices=(0.5, 1, 2),
                   help='a lower scale is faster but does not mean worse quality, default 1, try 0.5 for 4K')
    p.add_argument('-fp16', action='store_true', help='fp16 mode for faster inference on GPUs with Tensor Cores')
    e, c, q, s = ('NO', 'RLE', 'ZIPS', 'ZIP', 'PIZ', 'B44', 'B44A'), range(10), range(101), ('444', '422', '420')
    p.add_argument('-exr', default='B44A', type=str, choices=e, help='exr compression type, default B44A')
    p.add_argument('-png', default=1, type=int, choices=c, metavar='[0-9]', help='png compression, 0=off, default 1')
    p.add_argument('-jpg-q', default=98, type=int, choices=q, metavar='[0-100]', help='jpg quality, default 98')
    p.add_argument('-jpg-s', default='444', type=str, choices=s, metavar='4XX', help='jpg subsampling, default 444')
    p.add_argument('-mc', type=int, metavar='INT', help=f'(<v3.9) max number of splitting cycles, default {MAXC}')
    p.add_argument('-th', type=float, metavar='0-0.1', help=f'(<v3.9) done when ratio is close enough, default {THR}')
    args = p.parse_args()

    if args.r is not None and not 0 < args.r < 1:
        raise ValueError('-r should be in range (0-1)')
    write_kwargs = {'exr': args.exr, 'png': args.png, 'jpg_q': args.jpg_q, 'jpg_s': args.jpg_s}
    MAXC = args.mc if args.mc else MAXC
    THR = args.th if args.th else THR

    model_name, device = LoadModel(args.model, args.fp16)
    StartModel()
    print(f"Loaded {model_name} and using {device}")

    img0, h0, w0, a = ReadImage(*os.path.split(args.img[0]))
    img1, h, w, b = ReadImage(*os.path.split(args.img[1]))
    is_uint16 = a or b

    if h != h0 or w != w0:
        raise Exception('ERROR: dimensions of the 2 images are different')

    t = time.perf_counter()
    if args.r:
        img_list = [RatioSplit(img0, img1, args.r, args.scale)]
    else:
        img_list = MultiplyFPS(img0, img1, args.x, args.scale)
    print(f"Rendered in {time.perf_counter() - t:.3f} seconds")

    exts = os.path.splitext(args.img[0])[1].lower(), os.path.splitext(args.img[1])[1].lower()
    ext = '.tif' if '.tif' in exts or '.tiff' in exts else exts[0]
    ext = '.png' if '.png' in exts else ext
    ext = '.exr' if '.exr' in exts else ext
    [WriteImage('out', f'img_{i + 1}{ext}', img_list[i], h, w, is_uint16, write_kwargs) for i in range(len(img_list))]
    print('File(s) saved')


if __name__ == "__main__":
    main()
