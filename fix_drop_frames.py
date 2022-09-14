import concurrent.futures
import os
import re
import time
from collections import Counter, deque
from queue import Empty, Full, Queue

import numpy as np
import rawpy
from colorama import init, Fore
from tqdm import tqdm

from inference_img import LoadModel, StartModel, RatioSplit, ReadImage, WriteImage

init(autoreset=True)
G, R = Fore.LIGHTBLACK_EX, Fore.LIGHTRED_EX
T = "  "
H = G + T + "- "
ERR = R + T + "ERROR: "
WRN = Fore.YELLOW + T + "WARNING: "
HEAD = Fore.CYAN + "\n "
OUT = Fore.LIGHTGREEN_EX + T

MAX_DROP_LEN = 4
ERRORS = []
STOP_READ, STOP = False, False
APART, SCAN_ONLY = False, False
SCALE, WRITE_KWARGS, MODEL_NAME = [None] * 3
RE_NUM = re.compile(r"\d{3,}(?=\.[^.]+$)")
DROPPED_FRAMES_TXT = "_dropped_frames.txt"
RENDERED_TXT = "_rendered"
TRASH_DIR = "_Original files"
SUB_DIR = "_Rendered"


def detect_seq(files: list, ext: str | tuple) -> tuple:
    seq = [f for f in files if f.lower().endswith(ext)]
    fn = len(seq)
    if len(seq) < 5:
        return [], None

    # Leave only the largest sequence with 3+ digit number before .extension
    names = [(re.match(r'.*?(?=\d{3,}\.[^.]+$)', f), RE_NUM.search(f)) for f in seq]
    names = [(n.group(0), t.group(0)) for n, t in names if n and t]
    names = [(n, len(t)) for n, t in names]
    name = Counter(names).most_common(1)[0][0]
    seq = [f for f in seq if re.match(fr'^{name[0]}\d{{{str(name[1])}}}\.[^.]+$', f)]

    ext = ext[0] if type(ext) == tuple else ext
    return sorted(seq), (name[0] + '#' * name[1] + ext, ext, fn)


def sequence(dir: str) -> list:
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    if os.path.exists(os.path.join(dir, DROPPED_FRAMES_TXT)) and not SCAN_ONLY:
        with open(os.path.join(dir, DROPPED_FRAMES_TXT), 'r') as f:
            (f.readline(), f.readline())
            txt = [r.split(', ') for r in f.read().splitlines()]
        txt = [[s.strip() for s in r if s.strip()] for r in txt]
        files.extend([i for r in txt for i in r[1:-1] if i not in files])
        files.sort()

    exts = ['.exr', ('.tif', '.tiff'), '.png', ('.jpg', '.jpeg'), '.dng']
    pack = [detect_seq(files, ext) for ext in exts]
    longest_id = max(range(len(pack)), key=[len(seq[0]) for seq in pack].__getitem__)
    if len(pack[longest_id][0]) < 5:
        return None
    seq, extra = pack[longest_id]

    if ', ' in extra[0]:
        msg = f"Sequence name '{extra[0]}' contains ', ' - please rename it in {dir}"
        print(f"{ERR}{msg}")
        ERRORS.append(msg)
        return None
    first = int(RE_NUM.search(seq[0]).group(0))
    last = int(RE_NUM.search(seq[-1]).group(0))
    seq_len = last - first + 1
    if seq_len > len(seq):
        msg = f"{seq_len - len(seq)} file(s) missing from a sequence in {dir}"
        print(f"{ERR}{msg}")
        ERRORS.append(msg)
        return None
    if len(seq) < extra[2]:
        print(f"{WRN}{extra[2] - len(seq)} extra {extra[1]} file(s) present in {dir}")
    return seq


def find_drop_frames(dir: str, seq: list, atol: float = 0, max_workers: int = 3) -> list:
    print(f"{HEAD}Looking for dropped frames in {dir}")
    timer = {'start': time.perf_counter(), 'diff': []}
    is_dng = seq[0].lower().endswith('.dng')
    if is_dng:
        print(f"{WRN}Because it's a DNG sequence, dropped frames won't be fixed, only detected")

    txt_path = os.path.join(dir, DROPPED_FRAMES_TXT)
    if os.path.exists(txt_path) and not SCAN_ONLY:
        with open(txt_path, 'r') as f:
            (f.readline(), f.readline())
            txt = [r.split(', ') for r in f.read().splitlines()]
        print(f"{H}found {DROPPED_FRAMES_TXT} file")

        txt = [[s.strip() for s in r if s.strip()] for r in txt]
        dropped_frames = [{'img0': r[0], 'dropped': r[1:-1], 'img1': r[-1]} for r in txt if len(r) >= 3]

        if len(dropped_frames) > 0:
            n = sum([len(d['dropped']) for d in dropped_frames])
            print(f"{OUT}Got {n} dropped frames across {len(dropped_frames)} drops")
        else:
            print(f"{OUT}No dropped frames listed")
            return None
    else:
        if SCAN_ONLY:
            print(f"{H}scanning for max. changes in a sequence of {len(seq)} files")
        else:
            print(f"{H}searching dropped frames in a sequence of {len(seq)} files")
            print(f"{H}using fuzzy matching with atol={atol}" if atol > 0 else f"{H}using exact matching")

        dropped_frames, drop, opened = [], {}, 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exec:
            if not is_dng:
                full_pool = deque(seq)
                read_pool = deque([exec.submit(ReadImage, dir, full_pool.popleft(), to_torch=False) for _ in range(8)])

            prev_frame, prev_img, h0, w0, same_before = [None] * 5
            pbar = tqdm(seq, unit=' frames', colour='yellow')
            for frame in pbar:
                try:
                    if not is_dng:
                        if len(full_pool) > 0:
                            read_pool.append(exec.submit(ReadImage, dir, full_pool.popleft(), to_torch=False))
                        item = read_pool.popleft().result(timeout=2)
                    else:
                        item = rawpy.imread(os.path.join(dir, frame)).raw_image
                except Exception as e:
                    exec.shutdown(wait=False, cancel_futures=True)
                    pbar.close()
                    msg = f"While opening {frame} in folder {dir}"
                    tqdm.write(f"{ERR}{repr(e)}\n  ...{msg}")
                    ERRORS.append(msg)
                    break
                img, h, w, _ = item if not is_dng else (item, *item.shape, None)
                opened += 1

                t = time.perf_counter()
                if SCAN_ONLY:
                    img = img.astype(np.int16) if img.dtype == np.uint8 else img
                    img = img.astype(np.int32) if img.dtype == np.uint16 else img
                same_now = False
                if prev_frame:
                    if h != h0 or w != w0:
                        tqdm.write(f"{WRN}Skipping {prev_frame} & {frame} because their dimensions are different")
                    else:
                        if SCAN_ONLY:
                            tqdm.write(f"{G}{T}   {frame} - {np.amax(np.abs(img - prev_img))}")  # max difference
                        else:
                            if atol > 0:
                                if np.allclose(img, prev_img, rtol=0, atol=atol):
                                    tqdm.write('     ' + frame)
                                    same_now = True
                            elif np.array_equal(img, prev_img):  # check for exact match
                                tqdm.write('     ' + frame)
                                same_now = True
                        timer['diff'].append(time.perf_counter() - t)

                        if same_now and not same_before:
                            drop['img0'] = prev_frame
                            drop['dropped'] = [frame]
                        elif same_now and same_before:
                            drop['dropped'].append(frame)
                        elif not same_now and same_before:
                            drop['img1'] = frame
                            dropped_frames.append(drop)
                            drop = {}
                prev_frame = frame
                prev_img, h0, w0 = img, h, w
                same_before = same_now

        if opened == len(seq) and not SCAN_ONLY:
            if dropped_frames:
                text = [', '.join([d['img0'], *d['dropped'], d['img1']]) for d in dropped_frames]
                d_len, df_len = len(dropped_frames), [len(d['dropped']) for d in dropped_frames]
                n, m, a = sum(df_len), max(df_len), sum(df_len) / d_len
                txt = (f"Dropped {n}/{len(seq)} frames across {d_len} drops. Longest drop: {m}, avg: {a:.2f}."
                       + f" Compared with atol={atol}\n" + '  img0, <dropped frames>, img1:\n'
                       + '\n'.join(text))
                print(f"{OUT}Found {n} dropped frames across {d_len} drops")
            else:
                txt = f"No dropped frames found in {len(seq)} frames sequence. Compared with atol={atol}"
                print(f"{OUT}No dropped frames found")
            with open(txt_path, 'w') as f:
                f.write(txt)
        if opened < len(seq) and not SCAN_ONLY:
            print(f"{WRN}Not saving {DROPPED_FRAMES_TXT} to {dir} because sequence analysis didn't complete")

        if timer['diff']:
            print(f"{T}Scanning took {time.perf_counter() - timer['start']:.1f} seconds in {dir}")
        print_time()
    return None if SCAN_ONLY or is_dng else dropped_frames


def fix_drop_frames(dir: str, dropped_frames: list):
    print(f"{HEAD}Fixing dropped frames in {dir}")
    timer = {'start': time.perf_counter(), 'open': [], 'rend': [], 'save': []}
    global STOP_READ, STOP
    STOP_READ = False

    rendered = set()
    txt_path, v = os.path.join(dir, RENDERED_TXT), 1
    while os.path.exists(f'{txt_path} v{v}.txt'):
        with open(f'{txt_path} v{v}.txt', 'r') as f:
            f.readline()
            rendered.update([r.strip() for r in f.read().splitlines() if r.strip() and r.strip()[-1] != '/'])
        print(f"{H}found '{RENDERED_TXT} v{v}.txt' file")
        v += 1
    txt_path = f'{txt_path} v{v}.txt'

    to_render = []
    for i, drop in enumerate(dropped_frames, 1):
        full = drop['dropped']
        if len(full) > MAX_DROP_LEN:
            tqdm.write("{}. >> skipping [{} .. {}]:  {} frames > {} (-max argument)".format(f"{G}{'*'+str(i):>6}",
                       drop['dropped'][0], drop['dropped'][-1], len(drop['dropped']), MAX_DROP_LEN))
            continue
        drop['ratios'] = [(i + 1) / (len(full) + 1) for i, name in enumerate(full) if name not in rendered]
        drop['dropped'] = [name for name in full if name not in rendered]
        if drop['dropped']:
            drop['i'] = i
            to_render.append(drop)

    if not to_render:
        print(f"{OUT}No dropped frames left to fix in {dir}")
        return None

    class WriteBufferStopped(Exception):
        pass
    read_buffer = Queue(maxsize=2)
    write_buffer = Queue(maxsize=4)
    total = len([f for drop in to_render for f in drop['dropped']])
    rbar = tqdm(desc='Render', total=total, unit=' frames', position=0, colour='blue')
    sbar = tqdm(desc='Saving', total=total, unit='  files', position=1, colour='green')

    with concurrent.futures.ThreadPoolExecutor() as exec:
        try:
            read_thread = exec.submit(build_read_buffer, read_buffer, dir, to_render)
            write_thread = exec.submit(clear_write_buffer, write_buffer, dir, total, txt_path, v, sbar)
            for _ in to_render:
                if write_thread.done():
                    raise WriteBufferStopped
                item = read_buffer.get(timeout=20)
                if item is None:
                    break
                drop, img0, img1, h, w, is_uint16 = item

                d_len = len(drop['ratios'])
                t = time.perf_counter()
                result = RatioSplit(img0, img1, drop['ratios'], SCALE)
                t = time.perf_counter() - t
                [timer['rend'].append(t / d_len) for _ in range(d_len)]
                rbar.update(d_len)

                i = drop['i']
                for name, img in zip(drop['dropped'], result):
                    if write_thread.done():
                        raise WriteBufferStopped
                    i = str(i) + '.' if type(i) is int else ''
                    write_buffer.put((i, name, img, h, w, is_uint16), timeout=30)
        except WriteBufferStopped:
            pass
        except Empty:
            msg = f"Read buffer is empty for too long ({dir})"
            tqdm.write(f"{ERR}{msg}")
            ERRORS.append(msg)
        except Full:
            msg = f"Write buffer is full for too long ({dir})"
            tqdm.write(f"{ERR}{msg}")
            ERRORS.append(msg)
        except KeyboardInterrupt:
            tqdm.write(f"{R}{T}Keyboard Interrupt !!!")
            STOP = True
        except Exception as e:
            msg = f"While going through rendering loop in {dir}"
            tqdm.write(f"{ERR}{repr(e)}\n  ...{msg}")
            ERRORS.append(msg)
        finally:
            STOP_READ = True
            if not read_buffer.empty():
                read_buffer.get()
            if write_thread.running():
                write_buffer.put(None)
            timer['open'] = read_thread.result()
            timer['save'] = write_thread.result()
            sbar.clear()
            rbar.close()
            sbar.close()

    if timer['open'] and timer['rend'] and timer['save']:
        real = time.perf_counter() - timer['start']
        threads_sum = sum(timer['open'] + timer['rend'] + timer['save'])
        t = {k: (sum(v), sum(v) / len(v)) for k, v in timer.items() if k != 'start' and v}

        print(f"{OUT}Saved {len(timer['save'])} new frames to {os.path.join(dir, SUB_DIR) if APART else dir}")
        print(f"{T}Fixing took {real:.1f} seconds ({threads_sum:.1f} threads sum, x{threads_sum / real:.2f})")
        print(f"{G}{T}    Threads    Total, s   AVG, s")
        print(f"{T}   Open file   {t['open'][0]:7.2f}    {t['open'][1]/2:.3f}")
        print(f"{T}   Render      {t['rend'][0]:7.2f}    {t['rend'][1]  :.3f}")
        print(f"{T}   Save file   {t['save'][0]:7.2f}    {t['save'][1]  :.3f}")
    else:
        print(f"{OUT}No results have been made in {dir}")
    print_time()


def build_read_buffer(read_buffer: Queue, dir: str, dropped_frames: list):
    timer = []
    for drop in dropped_frames:
        if STOP_READ:
            return timer
        t = time.perf_counter()
        try:
            img0, h0, w0, a = ReadImage(dir, drop['img0'])
            img1, h, w, b = ReadImage(dir, drop['img1'])
            is_uint16 = a or b
        except Exception as e:
            msg = f"While opening {drop['img0']} or {drop['img1']}, not rendered: {', '.join(drop['dropped'])}'"
            tqdm.write(f"{ERR}{repr(e)}\n  ...{msg}")
            ERRORS.append(msg)
            continue
        timer.append(time.perf_counter() - t)
        if h != h0 or w != w0:
            tqdm.write(f"{WRN}Skipping {drop['img0']} & {drop['img1']} because their dimensions are different")
            continue
        if not STOP_READ:
            read_buffer.put((drop, img0, img1, h, w, is_uint16))
    if not STOP_READ:
        read_buffer.put(None)
    return timer


def clear_write_buffer(write_buffer: Queue, dir: str, to_write_len: int, txt_path: str, v: int, sbar: tqdm):
    timer = []
    f_start = True

    trash_path = os.path.join(dir, TRASH_DIR)
    os.makedirs(trash_path, exist_ok=True)
    if APART:
        sub_path = os.path.join(dir, SUB_DIR)
        os.makedirs(sub_path, exist_ok=True)

    for _ in range(to_write_len):
        item = write_buffer.get()
        if item is None:
            break
        i, name, img, h, w, is_uint16 = item

        pathA, pathB = os.path.join(dir, name), os.path.join(trash_path, name)
        if os.path.exists(pathA) and not os.path.exists(pathB):
            os.rename(pathA, pathB)
        if APART and v > 1:
            name_v = os.path.splitext(name)
            name_v = f'{name_v[0]}v{v}{name_v[1]}'
        write_name = name_v if APART and v > 1 else name
        write_dir = sub_path if APART else dir
        t = time.perf_counter()
        try:
            WriteImage(write_dir, write_name, img, h, w, is_uint16, WRITE_KWARGS)
        except Exception as e:
            msg = f"While saving {write_name} to {write_dir}"
            tqdm.write(f"{ERR}{repr(e)}\n  ...{msg}")
            ERRORS.append(msg)
            tqdm.write(f"{T}Stopping all activity in this folder... ({dir})")
            if not write_buffer.empty():
                write_buffer.get()
            return timer
        timer.append(time.perf_counter() - t)
        sbar.update()

        with open(txt_path, 'a') as f:
            if f_start:
                f.write(MODEL_NAME + '\n')
                f_start = False
            f.write(name + '\n')
        tqdm.write(f"{G}{i:>7}{Fore.RESET} {write_name}" + (f" {G}- saved to {SUB_DIR}" if APART else ''))
    return timer


def print_time(title: str = 'Current time'):
    print(f"{Fore.LIGHTMAGENTA_EX} {title}: {time.strftime('%H:%M:%S', time.localtime())}")


def main():
    global MAX_DROP_LEN, APART, SCAN_ONLY, SCALE, WRITE_KWARGS, MODEL_NAME
    import argparse
    p = argparse.ArgumentParser(description='Fix dropped frames in a sequence using RIFE model')
    p.add_argument('-dir', type=str, required=True, help='input folder (includes subfolders)')
    p.add_argument('-atol', default=0.0, type=float, metavar='INT/FLOAT',
                   help='absolute tolerance for matching, default 0 (exact)')
    p.add_argument('-max', default=MAX_DROP_LEN, type=int, choices=range(1, 32), metavar='[1-31]',
                   help=f'ignore longer drops of consecutive frames, default {MAX_DROP_LEN}')
    p.add_argument('-apart', action='store_true', help='save frames to a subfolder instead of the source folder')
    p.add_argument('-no-render', action='store_true', help='exit after analyzing and dumping dropped frames to txts')
    p.add_argument('-scan-only', action='store_true', help='just compare frames and print max differences')
    p.add_argument('-threads', default=3, type=int, choices=range(1, 9), metavar='[1-8]',
                   help='number of concurrent threads that open files for the comparison, default 3')
    p.add_argument('-model', type=str, help='folder with RIFE model files, default is the latest rife** or train_log')
    p.add_argument('-scale', default=1.0, type=float, choices=(0.5, 1, 2),
                   help='a lower scale is faster but does not mean worse quality, default 1, try 0.5 for 4K')
    p.add_argument('-fp16', action='store_true', help='fp16 mode for faster inference on GPUs with Tensor Cores')
    e, c, q, s = ('NO', 'RLE', 'ZIPS', 'ZIP', 'PIZ', 'B44', 'B44A'), range(10), range(101), ('444', '422', '420')
    p.add_argument('-exr', default='B44A', type=str, choices=e, help='exr compression type, default B44A')
    p.add_argument('-png', default=1, type=int, choices=c, metavar='[0-9]', help='png compression, 0=off, default 1')
    p.add_argument('-jpg-q', default=98, type=int, choices=q, metavar='[0-100]', help='jpg quality, default 98')
    p.add_argument('-jpg-s', default='444', type=str, choices=s, metavar='4XX', help='jpg subsampling, default 444')
    args = p.parse_args()

    args.atol = int(args.atol) if args.atol.is_integer() else args.atol
    MAX_DROP_LEN, APART, SCAN_ONLY, SCALE = args.max, args.apart, args.scan_only, args.scale
    WRITE_KWARGS = {'exr': args.exr, 'png': args.png, 'jpg_q': args.jpg_q, 'jpg_s': args.jpg_s}
    if args.png > 1:
        print(f"{WRN}PNG compression > 1 offers negligible further space savings with quite longer saving time")

    print_time('Time at start')
    print(f"{HEAD}Looking for sequences of 5+ files of supported formats [exr, tif, png, jpg, (dng)]")
    dirs = [args.dir]
    seq_dirs = []
    while dirs:
        subdirs = []
        for dir in dirs:
            seq = sequence(dir)
            if seq:
                seq_dirs.append((dir, seq))
            else:
                all_dirs = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
                subdirs.extend([os.path.join(dir, f) for f in all_dirs if f not in (TRASH_DIR, SUB_DIR)])
        dirs = subdirs
    if seq_dirs:
        print(f"{OUT}Found {len(seq_dirs)} sequence" + ('s' if len(seq_dirs) > 1 else ''))
    else:
        print(f"{OUT}Not found any sequence in {args.dir} or its subfolders")

    to_fix = []
    for dir, seq in seq_dirs:
        dropped_frames = find_drop_frames(dir, seq, args.atol, args.threads)
        if not dropped_frames:
            continue
        to_fix.append((dir, dropped_frames))

    if to_fix and not args.no_render:
        print(f"{HEAD}Getting ready for rendering...")
        t0 = time.perf_counter()
        MODEL_NAME, device = LoadModel(args.model, args.fp16)
        t1 = time.perf_counter()
        print(f"{H}using {device}")
        print(f"{H}loaded  {MODEL_NAME} in {t1 - t0:.1f} seconds")
        StartModel()
        t2 = time.perf_counter()
        print(f"{H}started {MODEL_NAME} in {t2 - t1:.1f} seconds ({t2 - t0:.1f} total)")
        print_time()

        for dir, dropped_frames in to_fix:
            if not STOP:
                fix_drop_frames(dir, dropped_frames)

    if ERRORS:
        print(f"{R}\n ALL COLLECTED ERRORS:")
        print('\n'.join([f" {i}. {e}" for i, e in enumerate(ERRORS, 1)]))


if __name__ == '__main__':
    main()
