import sys
import importlib
import subprocess


def get_version(pkg_name):
    try:
        module = importlib.import_module(pkg_name)
        return getattr(module, "__version__", "unknown")
    except ImportError:
        return "NOT INSTALLED"


def print_header(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def main():
    print_header("PYTHON")
    print(f"Python version: {sys.version}")

    print_header("CORE LIBRARIES")
    libs = [
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "safetensors",
    ]

    for lib in libs:
        print(f"{lib}: {get_version(lib)}")

    print_header("MULTIMODAL / INTERNVL DEPENDENCIES")
    libs = [
        "timm",
        "einops",
        "sentencepiece",
        "huggingface_hub",
    ]

    for lib in libs:
        print(f"{lib}: {get_version(lib)}")

    print_header("CUDA INFO (PYTORCH)")
    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version (torch): {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("Torch not available")

    print_header("PIP FREEZE (IMPORTANT)")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True,
                                text=True)
        print(result.stdout)
    except Exception as e:
        print("Failed to run pip freeze:", e)


if __name__ == "__main__":
    main()
