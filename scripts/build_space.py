"""Assemble the Hugging Face Space (the Reachy Mini app-store entry) into dist/space/ and audit it for secrets.

Only the package and its metadata go in — no tests, wiki, recordings, scripts or env files.

    python scripts/build_space.py && hf upload dAAAb/Reachy-Motion dist/space . --repo-type space
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "space"
COPY = ["pyproject.toml", "LICENSE", "NOTICE", "reachy_motion"]
SPACE_FILES = ROOT / "space"  # README.md with HF front-matter, index.html, style.css
SECRET_PATTERNS = [r"sk-[A-Za-z0-9_-]{20,}", r"sk_[a-f0-9]{30,}", r"hf_[A-Za-z0-9]{30,}", r"ghp_[A-Za-z0-9]{30,}",
                   r"AKIA[0-9A-Z]{16}", r"-----BEGIN [A-Z ]*PRIVATE KEY", r"/Users/", r"@gmail", r"192\.168\."]


def build() -> Path:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    for name in COPY:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, OUT / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".env*", ".DS_Store"))
        else:
            shutil.copy2(src, OUT / name)
    for f in SPACE_FILES.iterdir():
        shutil.copy2(f, OUT / f.name)
    return OUT


def audit(folder: Path) -> list[str]:
    """Secret patterns, env-like files, and every value of the local .env files must be absent."""
    local_values: list[tuple[str, str]] = []  # (key name, value) of every key / token / ID in the local env files
    for env in (ROOT / ".env", Path.home() / ".config" / "reachy_motion" / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = (x.strip() for x in line.split("=", 1))
                    if re.search(r"(KEY|TOKEN|SECRET|PASSWORD|_ID)$", k) and len(v) >= 8:
                        local_values.append((k, v))
    problems = []
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(folder)
        if re.search(r"(^|/)\.env|\.pem$|\.key$|id_rsa|secret|credential", str(rel), re.I):
            problems.append(f"suspicious file name: {rel}")
        text = p.read_bytes().decode("utf-8", "ignore")
        problems += [f"{rel}: matches {pat}" for pat in SECRET_PATTERNS if re.search(pat, text)]
        problems += [f"{rel}: contains the local value of {k}" for k, v in local_values if v in text]
    return problems


if __name__ == "__main__":
    out = build()
    problems = audit(out)
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    print(f"{len(files)} files in {out}")
    if problems:
        print("AUDIT FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("audit OK: no secrets, no env files, no local key values")
