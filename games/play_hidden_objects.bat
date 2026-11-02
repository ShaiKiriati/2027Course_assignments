@echo off
rem Starts the Hidden Object Game using the conda environment "sam3-objects".
rem Keep this file in the same folder as hidden_objects_app.py and segment_and_name_sam3.py.
setlocal
set "ENV_NAME=sam3-objects"
set "PYEXE="

rem If conda is installed somewhere unusual, put the full path to the environment's python.exe here:
rem set "PYEXE=C:\path\to\envs\sam3-objects\python.exe"

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

"%PYEXE%" "%~dp0hidden_objects_app.py"
if errorlevel 1 pause
