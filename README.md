# css535-project
Repo for the CSS535 Optical Flow CUDA project

## Pre-requisities
- VS 2022
- Windows 11
- CUDA 12.8.0

## To test profiler
- Open a `Visual Studio 2022 Developer Command Prompt`
- Navigate to ..\css535-project\OpticalFlow
- Run the following command to test your CUDA installation
```
$ nvcc -o test test.cu profiler.cu
```
- Run output
```
$ .\test.exe
```
- Expected Results:
```
$ .\test.exe
Elapsed time Host to Device for h_data and d_data: 7 ms
Memcpy Host->Device Speed: 13.9509 GB/s
Elapsed time Host to Device for a: 0 ms
Elapsed time Host to Device for b: 0 ms
addKernel elapsed time: 0.379904 ms
Elapsed time Device to Host for c: 0 ms
{1,2,3,4,5} + {10,20,30,40,50} = {11,22,33,44,55}
```
