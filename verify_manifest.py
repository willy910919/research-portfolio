"""Verify tracked public file hashes without requiring the private originals."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'public_manifest.json').read_text(encoding='utf-8'))
errors = []
for item in manifest['files']:
    path = root / item['path']
    if not path.is_file():
        errors.append(item['path'] + ': missing')
        continue
    content = path.read_bytes()
    if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
        errors.append(item['path'] + ': content differs')
if errors:
    raise SystemExit('\n'.join(errors))
print(f"PASS: {len(manifest['files'])} public files match their SHA-256 entries.")
print('The manifest itself is excluded from its own hashes; generated local outputs are not checked.')
