@echo off
rem Installs Veda's FA4 kernels into the Python that runs ComfyUI.
rem Usage: install_fa4.bat [path\to\python.exe] [--check]
setlocal
set "HERE=%~dp0"
set "PY="
if not "%~1"=="" if exist "%~1" (set "PY=%~1" & shift)
rem ComfyUI portable: <root>\python_embeded, nodes in <root>\ComfyUI\custom_nodes
if not defined PY if exist "%HERE%..\..\..\python_embeded\python.exe" set "PY=%HERE%..\..\..\python_embeded\python.exe"
rem ComfyUI Desktop and venv installs: <base>\.venv or <base>\venv
if not defined PY if exist "%HERE%..\..\.venv\Scripts\python.exe" set "PY=%HERE%..\..\.venv\Scripts\python.exe"
if not defined PY if exist "%HERE%..\..\venv\Scripts\python.exe" set "PY=%HERE%..\..\venv\Scripts\python.exe"
if not defined PY set "PY=python"
echo Using %PY%
"%PY%" "%HERE%install_fa4.py" %1 %2
pause
