call "Z:\Dev\Cotracker\.venv\Scripts\activate.bat"

REM Fix OpenBLAS warning for high-core count systems (32 cores, 64 threads)
set OPENBLAS_NUM_THREADS=8
set MKL_NUM_THREADS=8
set NUMEXPR_NUM_THREADS=8

python.exe "Z:\Dev\Cotracker\cotracker_nuke_app.py" --log-level DEBUG
