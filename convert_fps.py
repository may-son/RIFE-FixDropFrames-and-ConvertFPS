import concurrent.futures
import os
import re
import shutil
import time
from collections import Counter
from queue import Empty, Full, Queue

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

ERRORS = []
STOP_READ, STOP = False, False
SCALE, WRITE_KWARGS, SUFFIX, COPY_ONLY, RENAME, MODEL_NAME = [None] * 6
RE_NUM = re.compile(r"\d{3,}(?=\.[^.]+$)")


def detect_seq(files: list, ext: str | tuple) -> tuple:
    seq = [f for f in files if f.lower().endswith(ext)]
    fn = len(seq)
    if len(seq) < 3:
        return [], None, None

    # Leave only the largest sequence with 3+ digit number before .extension
    names = [(re.match(r'.*?(?=\d{3,}\.[^.]+$)', f), RE_NUM.search(f)) for f in seq]
    names = [(n.group(0), t.group(0)) for n, t in names if n and t]
    names = [(n, len(t)) for n, t in names]
    name = Counter(names).most_common(1)[0][0]
    seq = [f for f in seq if re.match(fr'^{name[0]}\d{{{str(name[1])}}}\.[^.]+$', f)]

    ext = ext[0] if type(ext) == tuple else ext
    return sorted(seq), [name[0], name[1], ext], fn


def sequence(dir: str) -> tuple:
    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    exts = ['.exr', ('.tif', '.tiff'), '.png', ('.jpg', '.jpeg')]
    pack = [detect_seq(files, ext) for ext in exts]
    longest_id = max(range(len(pack)), key=[len(seq[0]) for seq in pack].__getitem__)
    if len(pack[longest_id][0]) < 3:
        return None, None
    seq, form, fn = pack[longest_id]

    first = int(RE_NUM.search(seq[0]).group(0))
    last = int(RE_NUM.search(seq[-1]).group(0))
    seq_len = last - first + 1
    if seq_len > len(seq):
        msg = f"{seq_len - len(seq)} file(s) missing from a sequence in {dir}"
        print(f"{ERR}{msg}")
        ERRORS.append(msg)
        return None, None
    if len(seq) < fn:
        print(f"{WRN}{fn - len(seq)} extra {form[2]} file(s) present in {dir}")
    return seq, form


def convert_fps(dir: str, seq: list, form: list, x: float = None):
    dir_out = dir + SUFFIX
    print(f"{HEAD}Converting FPS from a sequence in {dir}, saving to: {dir_out}")
    timer = {'start': time.perf_counter(), 'open': [], 'rend': [], 'save': [], 'copy': []}

    form[0] = re.sub(r'\*', form[0], RENAME) if RENAME else form[0]
    form[1] = max(form[1], len(str(int(x * (len(seq) - 1)))))
    global STOP_READ, STOP
    STOP_READ = False

    frames_out, j = [], 0
    prev_name = None
    for i, name in enumerate(seq, -1):
        new_names, ratios = [], []
        while (1 / x * j) - i <= 1:
            new_name = f"{form[0]}{j:0{form[1]}}{form[2]}"
            if not os.path.exists(os.path.join(dir_out, new_name)):
                new_names.append(new_name)
                ratios.append((1 / x * j) - i)
            j += 1
        if new_names:
            part = {'img0': prev_name, 'img1': name, 'names': new_names, 'ratios': ratios}
            frames_out.append(part)
        prev_name = name

    if not frames_out:
        print(f"{OUT}All {j} output frames already exist in {dir_out}")
        return None

    class WriteBufferStopped(Exception):
        pass
    read_buffer = Queue(maxsize=2)
    write_buffer = Queue(maxsize=3)
    to_rend_len = len([f for part in frames_out for f in part['ratios'] if f < 1])
    to_save_len = len([f for part in frames_out for f in part['names']])
    rbar = tqdm(desc='Render', total=to_rend_len, unit=' frames', position=0, colour='blue', disable=not to_rend_len)
    sbar = tqdm(desc='Saving', total=to_save_len, unit='  files', position=1, colour='green')
    os.makedirs(dir_out, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor() as exec:
        try:
            read_thread = exec.submit(build_read_buffer, read_buffer, dir, frames_out)
            write_thread = exec.submit(clear_write_buffer, write_buffer, dir, to_save_len, sbar)
            for _ in frames_out:
                if write_thread.done():
                    raise WriteBufferStopped
                item = read_buffer.get(timeout=20)
                if item is None:
                    break
                part, img0, img1, h, w, is_uint16 = item

                result = []
                if img0 is not None:
                    ratios = [f for f in part['ratios'] if f < 1]
                    t = time.perf_counter()
                    result = RatioSplit(img0, img1, ratios, SCALE)
                    t = time.perf_counter() - t
                    [timer['rend'].append(t / len(ratios)) for _ in range(len(ratios))]
                    rbar.update(len(ratios))

                if len(result) < len(part['names']):
                    result.append(None)

                for name, img in zip(part['names'], result):
                    if write_thread.done():
                        raise WriteBufferStopped
                    write_buffer.put((part['img1'], name, img, h, w, is_uint16), timeout=30)
        except WriteBufferStopped:
            pass
        except Empty:
            msg = f"Read buffer is empty for too long ({dir})"
            tqdm.write(f"{ERR}{msg}")
            ERRORS.append(msg)
        except Full:
            msg = f"Write buffer is full for too long ({dir_out})"
            tqdm.write(f"{ERR}{msg}")
            ERRORS.append(msg)
        except KeyboardInterrupt:
            tqdm.write(f"{R}{T}Keyboard Interrupt !!!")
            STOP = True
        except Exception as e:
            msg = f"While going through rendering loop in {dir_out}"
            tqdm.write(f"{ERR}{repr(e)}\n  ...{msg}")
            ERRORS.append(msg)
        finally:
            STOP_READ = True
            if not read_buffer.empty():
                read_buffer.get()
            if write_thread.running():
                write_buffer.put(None)
            timer['open'] = read_thread.result()
            timer['save'] = write_thread.result()[0]
            timer['copy'] = write_thread.result()[1]
            sbar.clear()
            rbar.close()
            sbar.close()

    if timer['save'] or timer['copy']:
        real = time.perf_counter() - timer['start']
        threads_sum = sum(timer['open'] + timer['rend'] + timer['save'] + timer['copy'])
        t = {k: (len(v), sum(v), sum(v) / len(v)) if v else (0, 0, 0) for k, v in timer.items() if k != 'start'}

        print(f"{OUT}Saved/copied {len(timer['save']) + len(timer['copy'])} new frames to {dir_out}")
        print(f"{T}Processing took {real:.1f} seconds ({threads_sum:.1f} threads sum, x{threads_sum / real:.2f})")
        print(f"{G}{T}    Threads           Total, s   AVG, s")
        print(f"{T}   Open file - {t['open'][0]:<6} {t['open'][1]:7.2f}    {t['open'][2]:.3f}")
        print(f"{T}   Render    - {t['rend'][0]:<6} {t['rend'][1]:7.2f}    {t['rend'][2]:.3f}")
        print(f"{T}   Save file - {t['save'][0]:<6} {t['save'][1]:7.2f}    {t['save'][2]:.3f}")
        print(f"{T}   Copy file - {t['copy'][0]:<6} {t['copy'][1]:7.2f}    {t['copy'][2]:.3f}")
    else:
        print(f"{OUT}No files have been added to {dir_out}")
    print_time()


def read_image(dir, name, at_stake):
    t = time.perf_counter()
    try:
        img = ReadImage(dir, name)
    except Exception as e:
        msg = f"While opening {name}, not rendered: {', '.join(at_stake)}'"
        tqdm.write(f"{ERR}{repr(e)}\n  ...{msg}")
        ERRORS.append(msg)
        return None
    return img, time.perf_counter() - t


def build_read_buffer(read_buffer: Queue, dir: str, frames: list):
    timer = []
    prev_img, prev_name = None, None
    for part in frames:
        to_rend_len = len([f for f in part['ratios'] if f < 1])
        img0 = None
        if to_rend_len and part['img0'] is not None:
            if prev_img and part['img0'] == prev_name:
                img0 = prev_img
            else:
                img0 = read_image(dir, part['img0'], part['names'])
                if not img0:
                    continue
                timer.append(img0[1])

        if STOP_READ:
            return timer
        if not to_rend_len:
            img0, img1, h, w, is_uint16 = [None] * 5
        else:
            img1 = read_image(dir, part['img1'], part['names'])
            prev_img, prev_name = img1, part['img1']
            if not img1:
                continue
            timer.append(img1[1])

            img1, h, w, is_uint16 = img1[0]
            if part['img0'] is not None:
                img0, h0, w0, b = img0[0]
                is_uint16 = is_uint16 or b
                if h != h0 or w != w0:
                    tqdm.write(f"{WRN}Skipping {part['img0']} & {part['img1']} because their dimensions are different")
                    continue
        if not STOP_READ:
            read_buffer.put((part, img0, img1, h, w, is_uint16))
    if not STOP_READ:
        read_buffer.put(None)
    return timer


def clear_write_buffer(write_buffer: Queue, dir: str, to_save_len: int, sbar: tqdm):
    dir_out = dir + SUFFIX
    timer, timer_copy = [], []
    for _ in range(to_save_len):
        item = write_buffer.get()
        if item is None:
            break
        img1_name, name, img, h, w, is_uint16 = item
        try:
            if img is None:
                t = time.perf_counter()
                shutil.copy2(os.path.join(dir, img1_name), os.path.join(dir_out, name))
                timer_copy.append(time.perf_counter() - t)
            else:
                t = time.perf_counter()
                WriteImage(dir_out, name, img, h, w, is_uint16, WRITE_KWARGS)
                timer.append(time.perf_counter() - t)
        except Exception as e:
            msg = f"While saving {name} to {dir_out}"
            tqdm.write(f"{ERR}{repr(e)}\n  ...{msg}")
            ERRORS.append(msg)
            tqdm.write(f"{T}Stopping all activity in {dir} and {dir_out}")
            if not write_buffer.empty():
                write_buffer.get()
            return timer, timer_copy
        sbar.update()
    return timer, timer_copy


def print_time(title: str = 'Current time'):
    print(f"{Fore.LIGHTMAGENTA_EX} {title}: {time.strftime('%H:%M:%S', time.localtime())}")


def main():
    global SCALE, WRITE_KWARGS, SUFFIX, COPY_ONLY, RENAME, MODEL_NAME
    import argparse
    p = argparse.ArgumentParser(description='Convert FPS of a sequence to any arbitrary FPS using RIFE model')
    p.add_argument('-dir', type=str, required=True, help='input folder (includes subfolders)')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('-x', type=float, metavar='[0.3-32]', help='FPS multiplier, arbitrary number from 1/32 to 32')
    g.add_argument('-fps', nargs=2, type=float, metavar=('IN', 'OUT'), help='source and target FPS, int or float')
    p.add_argument('-rename', type=str, help='new base name for new files, use * to put old name in that place')
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

    x = int(args.x) if args.x is not None and args.x.is_integer() else args.x
    fps_in, fps_out = [int(i) if i.is_integer() else i for i in args.fps] if args.fps else (None, None)
    if x is not None and not (x > 0 and 1.04 <= max(x, 1 / x) <= 32):
        raise ValueError(f"{R}FPS multiplier should be in range [1/32-32] and outside of 4% near 1")
    if args.fps and not (fps_in > 0 and fps_out > 0 and 1.04 <= max(fps_out / fps_in, fps_in / fps_out) <= 32):
        raise ValueError(f"{R}In and out FPS should be > 0 and differ by at least 4%, but no more than 32 times")

    print_time('Time at start')
    SCALE = args.scale
    WRITE_KWARGS = {'exr': args.exr, 'png': args.png, 'jpg_q': args.jpg_q, 'jpg_s': args.jpg_s}
    SUFFIX = f"_{f'x{x}' if x else f'{fps_in}to{fps_out}'}fps"
    x = fps_out / fps_in if fps_in and fps_out else x
    COPY_ONLY = not 1 / x % 1
    if COPY_ONLY:
        print(f"{WRN}With current -x or -fps ratio, there is no need for AI rendering - only file copying is required")

    if args.png > 1:
        print(f"{WRN}PNG compression > 1 offers negligible further space savings with quite longer saving time")
    if args.rename:
        RENAME = re.sub(r"[^*\w `~!@#$%^&()_=+\[\]{};',.-]", '', args.rename)
        print(f"{HEAD}Using '{RENAME}' as the base name for all saved and copied files")

    print(f"{HEAD}Looking for sequences of 3+ files of supported formats [exr, tif, png, jpg]")
    dirs = [args.dir]
    seq_dirs = []
    while dirs:
        subdirs = []
        for dir in dirs:
            seq, form = sequence(dir)
            if seq and not dir.endswith(SUFFIX):
                seq_dirs.append((dir, seq, form))
            else:
                subdirs.extend([os.path.join(dir, f) for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))])
        dirs = subdirs
    if seq_dirs:
        print(f"{OUT}Found {len(seq_dirs)} sequence" + ('s' if len(seq_dirs) > 1 else ''))
    else:
        print(f"{OUT}Not found any sequence in {args.dir} or its subfolders")

    if seq_dirs:
        if not COPY_ONLY:
            print(f"{HEAD}Getting ready for rendering...")
            t0 = time.perf_counter()
            MODEL_NAME, device = LoadModel(args.model, args.fp16)
            t1 = time.perf_counter()
            print(f"{H}using {R if device != 'CUDA' else ''}{device}")
            print(f"{H}loaded  {MODEL_NAME} in {t1 - t0:.1f} seconds")
            StartModel()
            t2 = time.perf_counter()
            print(f"{H}started {MODEL_NAME} in {t2 - t1:.1f} seconds ({t2 - t0:.1f} total)")
            print_time()

        for dir, seq, form in seq_dirs:
            if not STOP:
                convert_fps(dir, seq, form, x)

    if ERRORS:
        print(f"{R}\n ALL COLLECTED ERRORS:")
        print('\n'.join([f" {i}. {e}" for i, e in enumerate(ERRORS, 1)]))


if __name__ == '__main__':
    main()
