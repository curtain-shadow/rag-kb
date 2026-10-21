@echo off
REM One-click local serving: llama.cpp + Qwen2.5-1.5B-Instruct GGUF (fp16)
REM Model path: D:\profession\models\gguf\qwen2.5-1.5b-instruct-fp16.gguf
REM OpenAI-compatible endpoint: http://127.0.0.1:8080/v1

set SERVER=D:\profession\llama.cpp\llama-server.exe
set MODEL=D:\profession\models\gguf\qwen2.5-1.5b-instruct-fp16.gguf
set PORT=8080

if not exist "%SERVER%" (
    echo [ERROR] llama-server.exe not found: %SERVER%
    exit /b 1
)
if not exist "%MODEL%" (
    echo [ERROR] GGUF model not found: %MODEL%
    exit /b 1
)

echo Starting local model server (llama.cpp, CUDA) ...
echo   model : %MODEL%
echo   endpoint: http://127.0.0.1:%PORT%/v1
echo Press Ctrl+C to stop.
"%SERVER%" -m "%MODEL%" --host 127.0.0.1 --port %PORT% -ngl 99 -c 4096 --alias qwen2.5-1.5b-instruct
