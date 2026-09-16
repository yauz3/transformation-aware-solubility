from pathlib import Path
import subprocess
import csv
import shutil

REPO = Path.cwd()
LIMIT_BYTES = 1_900_000_000
CHUNK_SIZE = "1500M"

skip_parts = {".git"}

oversized = []
for p in REPO.rglob("*"):
    if not p.is_file():
        continue
    if any(part in skip_parts for part in p.parts):
        continue
    if ".chunks" in p.parts:
        continue
    if p.name.endswith(".part"):
        continue
    size = p.stat().st_size
    if size > LIMIT_BYTES:
        oversized.append((size, p))

oversized.sort(reverse=True)

manifest_path = REPO / "manifests" / "oversize_file_chunks.tsv"

if not oversized:
    print("OK: no files larger than 1.9 GB were found.")
    manifest_path.write_text("original_path\tsize_bytes\tsha256\tchunk_dir\n", encoding="utf-8")
    raise SystemExit(0)

rows = []
for size, p in oversized:
    rel = p.relative_to(REPO).as_posix()
    print(f"Splitting: {rel} ({size / 1024**3:.2f} GB)")

    sha = subprocess.check_output(["sha256sum", str(p)], text=True).split()[0]

    chunk_dir = p.parent / f"{p.name}.chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    prefix = chunk_dir / f"{p.name}.part-"
    subprocess.check_call([
        "split",
        "-b", CHUNK_SIZE,
        "-d",
        "-a", "3",
        "--additional-suffix=.part",
        str(p),
        str(prefix)
    ])

    chunk_rel = chunk_dir.relative_to(REPO).as_posix()
    rows.append({
        "original_path": rel,
        "size_bytes": str(size),
        "sha256": sha,
        "chunk_dir": chunk_rel,
    })

    p.unlink()
    print(f"  -> removed original oversized file: {rel}")
    print(f"  -> chunks written to: {chunk_rel}")

with manifest_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["original_path", "size_bytes", "sha256", "chunk_dir"],
        delimiter="\t"
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote manifest: {manifest_path.relative_to(REPO)}")
print("Done.")
