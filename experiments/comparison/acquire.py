"""Acquire only the five pinned CSV versions; changed bytes are never accepted."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'data/comparison-input-manifest.json'
SNAPSHOT_SHA256 = '8e0c3a5f3ce5829089fe5a4452bb86bed34c5b711edc4840526e18886ae137ce'


def safe(root, name):
    """Reject absolute/traversing names and symlinks at every component."""
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('Unsafe input path')
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Input symlink refused')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Input path escapes root')
    return path


def check(data, item):
    """Check both length and exact version before writing or using an input."""
    if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
        raise ValueError('Input version differs: ' + item['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--from-dir', type=Path)
    group.add_argument('--archive', type=Path)
    group.add_argument('--download', action='store_true')
    group.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    items = json.loads(MANIFEST.read_text())['files']
    pending = []
    archive = None
    if args.archive:
        if hashlib.sha256(args.archive.read_bytes()).hexdigest() != SNAPSHOT_SHA256:
            raise ValueError('Exact input snapshot required')
        archive = tarfile.open(args.archive, 'r:gz')
        for member in archive.getmembers():
            safe(ROOT, member.name)
            if not member.isfile() or member.linkname:
                raise ValueError('Only regular archive members are permitted')
    try:
        for item in items:
            destination = safe(ROOT, item['path'])
            if args.verify:
                data = destination.read_bytes()
            elif args.from_dir:
                data = safe(args.from_dir.resolve(), item['path']).read_bytes()
            elif archive:
                data = archive.extractfile(item['path']).read()
            else:
                with urllib.request.urlopen(item['url'], timeout=90) as response:
                    data = response.read()
            check(data, item)
            if destination.exists():
                check(destination.read_bytes(), item)
            else:
                pending.append((destination, data))
        # Validate every source and existing destination before changing any file.
        for destination, data in pending:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    finally:
        if archive:
            archive.close()
    print(json.dumps({'verified_inputs': len(items), 'bytes': sum(x['bytes'] for x in items)}))


if __name__ == '__main__':
    main()
