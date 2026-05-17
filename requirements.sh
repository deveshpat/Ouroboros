python -m pip install --upgrade pip
# CPU-only torch: ~200MB vs ~2.5GB for CUDA build — coordinator never uses GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu
# kaggle>=1.8.4 required: --accelerator flag for kernels push added in v1.8.4 (#907).
# The prior 403 (Session 15) was on KernelsApiService/GetKernel (pull endpoint).
# We use push-only; push endpoint is not blocked. kaggle>=1.8.3 gRPC is fine here.
pip install numpy "kaggle>=1.8.4" huggingface_hub safetensors requests wandb pytest peft