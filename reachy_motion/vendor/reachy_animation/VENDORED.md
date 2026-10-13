Vendored from https://github.com/pham-tuan-binh/reachy-animation at commit
90e25b5773353213cffe18d915d0703ba04c083b (Apache-2.0, see LICENSE in this folder).

Only the runtime modules are included (animator, clips, motion, pose, speech); `sim.py` (MuJoCo rendering) stays
in the upstream package, installed with the `record` extra. Changes: imports rewritten to this package path, so the
app installs without git (Reachy Mini Control installs Space apps with `uv pip`; a git dependency needs a working
`git`, which macOS blocks until the Xcode license is accepted).
