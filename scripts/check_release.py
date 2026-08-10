#!/usr/bin/env python
"""Fail when review placeholders remain in a claimed paper-exact release."""
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument('--root', default='.')
p.add_argument('--allow-review-defaults', action='store_true')
p.add_argument('--review-stage', action='store_true', help='Permit all explicitly marked review placeholders')
a = p.parse_args()
root = Path(a.root)
needles = [] if a.review_stage else ['AUTHOR_REQUIRED', 'OWNER/TiDAL-Net']
if not (a.allow_review_defaults or a.review_stage):
    needles.append('REVIEW_DEFAULT')
ignore = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache'}
hits = []
for path in root.rglob('*'):
    if not path.is_file() or any(part in ignore for part in path.parts):
        continue
    if path.suffix.lower() in {'.pt', '.pth', '.npz', '.npy', '.pdf', '.png', '.jpg', '.zip', '.gz'}:
        continue
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        continue
    for needle in needles:
        for line_no, line in enumerate(text.splitlines(), 1):
            if needle in line:
                hits.append((str(path), line_no, needle))
if hits:
    print('Release check failed:')
    for path, line, needle in hits[:100]:
        print(f'  {path}:{line}: {needle}')
    raise SystemExit(1)
print('Release check passed.')
