@echo off
rem Starts the CLIP label-training interface without leaving a terminal window open.
rem Keep this file in the same folder as hidden_objects_web.py and its model modules.
setlocal
set "ENV_NAME=web-obj-game"
set "PYEXE="

rem If conda is installed somewhere unusual, put the full path to the environment's python.exe here:
rem set "PYEXE=C:\path\to\envs\web-obj-game\python.exe"

if not defined PYEXE (
  for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "C:\ProgramData\anaconda3" "C:\ProgramData\miniconda3") do (
    if exist "%%~D\envs\%ENV_NAME%\python.exe" if not defined PYEXE set "PYEXE=%%~D\envs\%ENV_NAME%\python.exe"
  )
)

if not defined PYEXE (
  echo Could not find the conda environment "%ENV_NAME%".
  echo Open an Anaconda Prompt, type:  conda activate %ENV_NAME%
  echo then type:  where python
  echo and put the first path it prints into the PYEXE line near the top of this file.
  pause
  exit /b 1
)

for %%F in ("%PYEXE%") do set "PYWEXE=%%~dpFpythonw.exe"
if not exist "%PYWEXE%" (
  echo Could not find pythonw.exe next to "%PYEXE%".
  pause
  exit /b 1
)

start "" "%PYWEXE%" "%~dp0hidden_objects_web.py" --training --port 5001
