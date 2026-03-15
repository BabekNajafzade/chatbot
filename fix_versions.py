import importlib
import os

packages = [
    "beautifulsoup4",
    "faiss",
    "numpy",
    "openai",
    "pandas",
    "python_dotenv",      
    "telegram",         
    "rank_bm25",
    "requests",
    "tqdm"
]

req_file = "requirements.txt"
lines = []

for pkg in packages:
    try:
        module = importlib.import_module(pkg)
        version = getattr(module, "__version__", "Unknown")
        if version != "Unknown":
            lines.append(f"{pkg.replace('_', '-') }=={version}")
        else:
            lines.append(f"{pkg.replace('_', '-')}")
    except ModuleNotFoundError:
        print(f"[!] {pkg} quraşdırılmayıb, requirements.txt-də qeyd edilməyəcək")

with open(req_file, "w") as f:
    f.write("\n".join(lines))

print(f"\n{req_file} yaradıldı/yeniləndi, daxil olan paketlər:")
for line in lines:
    print(" -", line)