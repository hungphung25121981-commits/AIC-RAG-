
#python checkenv.py dùng này để check
#python check_env.py --strict dùng trong CI/CD nếu mà thiếu thì dừng lại


import argparse
import importlib
import sys
import warnings

# Bỏ qua các SyntaxWarning coi cho dễ
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Danh sách các package cần kiểm tra trong pipeline
PACKAGES = [
    ("transformers", "Transformers"),
    ("accelerate", "Accelerate"),
    ("bitsandbytes", "BitsAndBytes"),
    ("safetensors", "SafeTensors"),
    ("sentence_transformers", "SentenceTransformers"),
    ("FlagEmbedding", "FlagEmbedding"),
    ("faiss", "FAISS"),
    ("rank_bm25", "Rank BM25"),
    ("rapidocr_onnxruntime", "RapidOCR"),
    ("onnxruntime", "ONNX Runtime"),
    ("cv2", "OpenCV"),
    ("scenedetect", "PySceneDetect"),
    ("skimage", "Scikit-Image"),
    ("PIL", "Pillow"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("pyarrow", "PyArrow"),
    ("yaml", "PyYAML"),
]

# Optional -- chỉ cần nếu bật phase2.vlm_use_qlora: true trong config/settings.yaml
OPTIONAL_PACKAGES = [
    ("peft", "PEFT (QLoRA adapter cho Qwen2.5-VL-3B)"),
]


def check_package(module_name, display_name):
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "Đã cài đặt (Không rõ version)")
        print(f"  [OK]  {display_name:<25} | Version: {version}")
        return True
    except ImportError:
        print(f"  [MISSING] {display_name:<21} | Chưa cài đặt!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Tool kiểm tra môi trường Video-Visual-RAG CLI")
    parser.add_argument("--strict", action="store_true", help="Trả về exit code 1 nếu có package thiếu")
    args = parser.parse_args()

    print("=" * 65)
    print(" CHECKING SYSTEM & DEPENDENCY ENVIRONMENT ")
    print("=" * 65)

    all_passed = True
    for module_name, display_name in PACKAGES:
        status = check_package(module_name, display_name)
        if not status:
            all_passed = False

    print("-" * 65)
    print(" Optional (QLoRA):")
    for module_name, display_name in OPTIONAL_PACKAGES:
        check_package(module_name, display_name)

    print("=" * 65)
    if all_passed:
        print(" Mọi thư viện đã sẵn sàng!")
        sys.exit(0)
    else:
        print(" Có thư viện chưa cài đặt. Vui lòng kiểm tra lại requirements.txt!")
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
