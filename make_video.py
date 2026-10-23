"""
Build the submission video from the two camera recordings of fig8_eval.py:
title card (with the playback speed) -> overview | follow side by side -> result plot.

python make_video.py --out-dir outputs --speed 1
"""

import argparse
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 540


def card(path, lines, size=(W, H)):
    img = Image.new("RGB", size, (16, 24, 40))
    draw = ImageDraw.Draw(img)
    y = (size[1] - sum(int(font_size * 1.6) for _, font_size in lines)) / 2
    for text, font_size in lines:
        font = ImageFont.load_default(size=font_size)
        w = draw.textlength(text, font=font)
        draw.text(((size[0] - w) / 2, y), text, fill=(240, 240, 240), font=font)
        y += int(font_size * 1.6)
    img.save(path)


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed of the main part (1 or 2)")
    parser.add_argument("--title", type=str, default="Figure-eight walking of a quadruped robot (Go2)")
    args = parser.parse_args()
    o = lambda name: os.path.join(args.out_dir, name)  # noqa: E731

    speed_text = "1x speed (real time)" if args.speed == 1 else f"{args.speed:g}x speed"
    card(
        o("title.png"),
        [
            (args.title, 64),
            ("Genesis simulator / RL locomotion policy (PPO) + pure-pursuit path following", 36),
            ("left: overview camera   right: follow camera   orange: reference path   blue: robot trail", 30),
            (speed_text, 44),
        ],
    )

    # 3 s title card
    run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", o("title.png"), "-t", "3", "-r", "25",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", o("part_title.mp4")])
    # side by side main part (+ speed up)
    run(["ffmpeg", "-y", "-v", "error", "-i", o("overview.mp4"), "-i", o("follow.mp4"), "-filter_complex",
         f"[0:v]scale=960:540[a];[1:v]scale=960:540[b];[a][b]hstack=inputs=2,setpts=PTS/{args.speed},fps=25[v]",
         "-map", "[v]", "-pix_fmt", "yuv420p", "-c:v", "libx264", o("part_main.mp4")])
    parts = [o("part_title.mp4"), o("part_main.mp4")]
    # 5 s result plot
    if os.path.exists(o("fig8_result.png")):
        run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", o("fig8_result.png"), "-t", "5", "-r", "25",
             "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:white",
             "-pix_fmt", "yuv420p", "-c:v", "libx264", o("part_result.mp4")])
        parts.append(o("part_result.mp4"))

    with open(o("parts.txt"), "w") as f:
        f.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", o("parts.txt"), "-c", "copy",
         o("fig8_submission.mp4")])
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          o("fig8_submission.mp4")], capture_output=True, text=True).stdout.strip()
    print(f"saved {o('fig8_submission.mp4')} ({float(out):.1f} s; the assignment requires 60-180 s)")


if __name__ == "__main__":
    main()
