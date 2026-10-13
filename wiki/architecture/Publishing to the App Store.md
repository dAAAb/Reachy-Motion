---
title: Publishing to the App Store
date: 2026-10-13
tags: [deployment, hugging-face, app-store]
aliases: [HF Space, Discover apps]
---

# Publishing to the Reachy Mini app store (done 2026-10-13)

Space: **https://huggingface.co/spaces/dAAAb/reachy_motion** (public). Build + audit + upload:

```bash
python scripts/build_space.py        # dist/space = package + space/README.md (front-matter) + index.html + public/icon.svg
hf upload dAAAb/reachy_motion dist/space . --repo-type space
reachy-mini-app-assistant check dist/space   # official validator: passes
```

The build copies only the package and its metadata and fails on secret patterns, env-like files, or any local
key/ID value. Keys are entered per device on the settings page (`userconfig.py`, write-only secrets).

## How listing works `(source code, verified live)`
- **Discover apps = every Space tagged `reachy_mini`** via the website catalog
  `pollen-robotics-reachy-mini.hf.space/api/apps` (cache ≈ 5 min). Default view shows community + official,
  sorted by **likes** — new apps start at the bottom. "Official" is a filter checkbox.
- **Official badge** = being in `pollen-robotics/reachy-mini-official-app-store/app-list.json`, via a PR
  (`reachy-mini-app-assistant publish --official` opens it). As of 2026-10 the PR queue had not been merged since
  2025-12. Not requested yet.
- Tile icon: `public/icon.svg` (or `.png`) in the Space; otherwise the front-matter emoji.

## Gotchas
- **Space name must equal the entry-point name** (`reachy_motion`): the shared-venv installed-app list looks up
  install metadata by exact name, so a Space called `Reachy-Motion` installed fine but never showed in Control's
  Applications list. Renamed with `HfApi().move_repo` (old URL redirects).
- **No git dependencies**: Control installs with `uv pip`; on a Mac whose `/usr/bin/git` is blocked by an
  un-accepted Xcode license (exit 69) a `git+https` dependency fails. `reachy-animation` is vendored.
- Installs done outside the UI (API / pip) only appear in Control after it re-fetches — restart Control.
- `short_description` ≤ 60 characters (Hub validator).

Related: [[Running on the Robot]], [[Body Agent]].
