from pathlib import Path

def hash_files(files: list[Path]):
    import hashlib
    hash_md5 = hashlib.md5()
    for file_path in sorted(files):
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()