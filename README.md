# AI drop frame fixer and FPS converter
**For image sequence from video. Not limited to 8 bits! Not restricted to standard FPS values!**

Using [Practical RIFE](https://github.com/hzwer/Practical-RIFE) AI models. Intended for creators. Works only with sequences of images (exr, tif, png, jpg). Can work through all sequences found in all subdirectories of the specified input folder. All scripts will save files in the same format as the input files. So even if provided arguments for jpeg quality, but input is png - it will save png.

## Dropped frame fixer: `fix_drop_frames.py`
Find dropped (skipped) frames in a sequence and create new interpolated frames in place. By dropped frame, I mean the same frame as the previous, pixel by pixel. Not a missing file, not identical checksums. Dropped frames won't be replaced, just moved to a "_Original files" subfolder, but if there is already a file with a given name (from previous runs) - it will replace files in the sequence folder (if not used `-apart`).
* "_dropped_frames.txt" is for storing found dropped frames.
* "_rendered v*.txt" is for storing already rendered and saved new frames, so the program can continue from the last file in case of a crash or outage.
* Use `-max` to set a custom number of maximum consecutive dropped frames to fix. Longer streaks would be ignored.
* Use `-apart` to render all frames separately and put them in a "_Rendered" subfolder instead of the sequence folder.
* After the first run, you can open "_rendered v1.txt" and add `/` at the end of frames you want to rerender (maybe with another RIFE model), and run again. You can repeat by adding `/` to every frame that you want to render again in the last "_rendered v*.txt" file. `/` at the end is the same as removing a filename entirely. Recommended to use `-apart` for that so new versions will be saved with version suffix instead of replacing existing files.
* If there is a sequence of **dng** files, the program will only find dropped frames without fixing them. Also, you can force that for all formats using `-no-render` option.
* You can also specify a tolerance for how much some pixel needs to differ to say that this frame is different using `-atol`. To know what to set, use `-scan-only` option to print out max differences for every pair of neighboring frames.

## FPS converter: `convert_fps.py`
Converts from one arbitrary FPS to another. For example, you can convert from 15 to 24, or 42.55 to 28.2. It takes a folder with a sequence inside and creates a new folder next to it with the same name plus added suffix showing FPS multiplier or in/out FPS.

Note that if the ratio is uneven (not x2, x3, etc.), some original frames will not make it into the new sequence. So don't delete the original if you need them. Also, it copies files from the original folder when needed instead of resaving. So FPS from metadata will not change. Conform to the required FPS in your editor manually every time.

## How is sequence detected:
* All files should have the same name, extension, and amount of digits for the number
* Should have at least 3 digit frame number in the name
* No frame number can be skipped
* Immediately after the number should be .extension
* One sequence per folder, in other words: folder = video

### Also: improved `inference_img.py`
Interpolate between two frames with a specified FPS multiplier or ratio. Based on code from [Practical RIFE](https://github.com/hzwer/Practical-RIFE). I modified it for my needs. This file is required by others to work.

## Usage:
_Note: this program needs PyTorch, which is a few GB in size._

_Also, having GPU is recommended (with the GPU version of PyTorch that you need to install manually using their [website](https://pytorch.org/get-started) before installing requirements.txt)._

Install Python, download zip from this repository and unpack, then navigate to the unpacked folder in the command line terminal, and run:
```
pip3 install -r requirements.txt
```

Run `py convert_fps.py -h` (or another .py file with `-h` argument) and read what arguments are available and what they do, then write what you need.

### Examples:
Fix dropped frames in a sequence in **D:\Forest**:
```
py fix_drop_frames.py -dir "D:\Forest"
```
Fix dropped frames in all sequences in **D:\Footage\MotionCamEXR** using RIFE v4.3, but ignore streaks of consecutive dropped frames longer than 2. Also, put new files in a subfolder (apart) for easy examination and further action (like rerendering them with another model):
```
py fix_drop_frames.py -dir "D:\Footage\MotionCamEXR" -model rife43 -max 2 -apart
```
---
Make x3.2 slow-motion for a sequence in **D:\Cat fall**:
```
py convert_fps.py -dir "D:\Cat fall" -x 3.2
```
Convert all sequences in **E:\Footage\25fps** from 25 to 29.97 FPS, add "**new_**" to the beginning of all new filenames, and use model v4.1:
```
py convert_fps.py -dir "E:\Footage\25fps" -fps 25 29.97 -rename "new_*" -model rife41
```

## Extra info:
To integrate new models, if they are still backward compatible:
1. Unpack the "train_log" folder to the program folder.
2. Rename it to "rifeXX" (version without dot) and change all "train_log" occurrences in .py files to the new folder name.
3. In "RIFE_HDv3.py", change the **self.version** number to the relevant actual version if the author didn't update it.

You can use train_log without changing anything, but then manually specify `-model train_log` or remove all "rifeXX" folders, so the program will fall back to using train_log.

### History
I actively worked on this project from 2022-08-07 to 2022-09-04.

* `fix_drop_frames.py` developed from 0 to "v1" in 21 days (2022-08-09 - 2022-08-29)
* `convert_fps.py` developed from 0 to "v1" in 4 days (2022-08-30 - 2022-09-02) _(with some copypasting from previous)_
