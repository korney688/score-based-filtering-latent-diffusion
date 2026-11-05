"""Build a Docker context from source files."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    if sys.argv[1] == '--send':
        with Path(sys.argv[2]).open('rb') as handle:
            raise SystemExit(subprocess.run(['docker', *sys.argv[3:]], stdin=handle).returncode)
    output = Path(sys.argv[1])
    paths = [ROOT / name for name in ('Dockerfile.noise-replicates', 'requirements.txt')]
    for folder, suffixes in (('src', {'.py'}), ('scripts', {'.py', '.sh', '.ps1'}), ('configs', {'.yaml', '.yml', '.json'})):
        paths.extend(p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts
                     and 'tests' not in p.parts and not p.name.startswith(('test_', 'smoke_'))
                     and not p.name.endswith('_test.py'))
    manifest = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w') as archive:
        for path in sorted(paths):
            if path.is_symlink():
                raise RuntimeError(f'Symlink in build context: {path}')
            name = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            # Use LF line endings for Linux shell scripts.
            data = data.replace(b'\r\n', b'\n')
            manifest[name] = hashlib.sha256(data).hexdigest()
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))
        data = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
        entry = tarfile.TarInfo('SOURCE_MANIFEST.json')
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    print(hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
