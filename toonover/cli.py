"""Command line.

  toonover make <link-or-file> [--style cartoon]   real video -> cartoon of the same people
  toonover photo <image> [--style cartoon]          quick test on one picture
  toonover make <link> -s ghibli                     GPU redraw (needs ComfyUI running)
  toonover styles                                   list styles
  toonover serve                                    web app
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .redraw.comfy import presets as redraw_presets
from .stylize import ALL_HELP, STYLES


def _progress_printer():
    last = {"s": None}

    def prog(stage, f):
        if stage != last["s"]:
            if last["s"] is not None:
                print()
            last["s"] = stage
        print(f"\r[{stage:<10}] {f * 100:5.1f}%", end="", flush=True)

    return prog


ALL_STYLES = [*redraw_presets(), *STYLES]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="toonover", description="Turn real videos into cartoons of the same people.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    mk = sub.add_parser("make", help="convert a video link or file into a cartoon")
    mk.add_argument("source", help="YouTube/Facebook/Instagram/TikTok link or a local video file")
    mk.add_argument("-s", "--style", default="ghibli", choices=ALL_STYLES, help="see 'toonover styles'")
    mk.add_argument("-o", "--out", type=Path, help="output .mp4 (default: output/)")
    mk.add_argument("-t", "--title", default="", help="add a black title bar on top with this text")
    mk.add_argument("--auto-title", action="store_true", help="use the video's own title for the bar")
    mk.add_argument("--channel", default="", help="add a bottom bar with this text (e.g. Toonover)")
    mk.add_argument("--max-seconds", type=float, default=90)
    mk.add_argument("--ai-size", type=int, default=512, help="AI styles: processing size (smaller = faster)")
    mk.add_argument("--ai-every", type=int, default=2, help="AI styles: run model every Nth frame (2 = 'on twos')")
    mk.add_argument("--keyframe-every", type=int, default=None, help="GPU styles: redraw every Nth frame (default 10)")
    mk.add_argument("--describe", default="", help="GPU styles: extra details, e.g. 'woman, long brown hair, gold hoop earrings'")
    mk.add_argument("--comfy-url", default=None, help="GPU styles: ComfyUI address (default http://127.0.0.1:8188)")

    ph = sub.add_parser("photo", help="cartoonize a single image (quick style test)")
    ph.add_argument("image", type=Path)
    ph.add_argument("-s", "--style", default="ghibli", choices=ALL_STYLES)
    ph.add_argument("-o", "--out", type=Path)
    ph.add_argument("--describe", default="", help="GPU styles: extra details (hair colour, clothes ...)")

    tn = sub.add_parser("tune", help="GPU styles: try several settings on one picture and save a comparison sheet")
    tn.add_argument("image", type=Path)
    tn.add_argument("-s", "--style", default="ghibli")
    tn.add_argument("--describe", default="")

    sub.add_parser("styles", help="list styles")

    sv = sub.add_parser("serve", help="start the web app")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)

    a = ap.parse_args(argv)

    if a.cmd == "make" and a.style in redraw_presets():
        from .convert import convert_redraw
        out = convert_redraw(a.source, a.style, a.out, title=a.title, channel=a.channel,
                             max_seconds=a.max_seconds, keyframe_every=a.keyframe_every,
                             comfy_url=a.comfy_url, auto_title=a.auto_title, describe=a.describe,
                             log=lambda m: print(f"\n  {m}"), progress=_progress_printer())
        print(f"\nSaved -> {out}")

    elif a.cmd == "make":
        from .convert import convert_video
        out = convert_video(a.source, a.style, a.out, title=a.title, channel=a.channel,
                            max_seconds=a.max_seconds, ai_size=a.ai_size, ai_every=a.ai_every,
                            auto_title=a.auto_title, log=lambda m: print(f"\n  {m}"),
                            progress=_progress_printer())
        print(f"\nSaved -> {out}")

    elif a.cmd == "photo":
        from .convert import convert_image
        out = a.out or a.image.with_name(f"{a.image.stem}_{a.style.replace('+', '-')}.png")
        print(f"Saved -> {convert_image(a.image, a.style, out, describe=a.describe)}")

    elif a.cmd == "tune":
        from .redraw.tune import tune
        print(f"Saved -> {tune(a.image, a.style, describe=a.describe)}")

    elif a.cmd == "styles":
        for s in ALL_STYLES:
            print(f"  {s:<14} {ALL_HELP.get(s, 'AI style + cel/ink finish')}")

    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("toonover.web.app:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
