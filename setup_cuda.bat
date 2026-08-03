@echo off
echo ============================================
echo  CUDA Setup for 4K Image Upscaler
echo  Requires Python 3.12 and NVIDIA drivers
echo ============================================
echo.

py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python 3.12 not found.
    echo Download it from https://python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] Creating Python 3.12 virtual environment...
py -3.12 -m venv venv
call venv\Scripts\activate

echo [2/5] Installing PyTorch with CUDA 12.4...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo [3/5] Installing basicsr...
pip install basicsr

echo [4/5] Installing Real-ESRGAN and dependencies...
pip install realesrgan facexlib gfpgan

echo [5/5] Installing remaining packages...
pip install opencv-python Pillow tqdm gradio einops

echo.
echo ============================================
echo  Done! Verify GPU is detected:
echo    python -c "import torch; print(torch.cuda.get_device_name(0))"
echo.
echo  Then launch:
echo    python app.py
echo ============================================
pause
