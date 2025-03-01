# css535-project
Repo for the CSS535 Optical Flow CUDA project

## Pre-requisities
- VS 2022
- Windows 11
- CUDA 12.8.0
- Rerun 

## To test profiler
- Open a `Visual Studio 2022 Developer Command Prompt`
- Navigate to ..\css535-project\OpticalFlow

## To build and run
Navigate to the OpticalFlow folder then in a VS 2022 Developers CMD:
```
nmake /f MakeFile
.\test.exe
```
#### Alternately
```
nvcc -o test test.cu profiler.cu
.\test.exe
```
### Expected result:
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

## To clean files
Navigate to the OpticalFlow folder then in a VS 2022 Developers CMD:
```
nmake clean
```

## Pixi package management
- Pixi is a package management tool for developers and allows developers to install libraries and applications in a reproducible way.

## Installing Pixi
```
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

## Building the test main.cpp
From within the project root directory
```
pixi run start
```

## Rerun
A multimodal data logger and visualizer

## Setting up Rerun
### Web Browser
Open this in a web browser : https://rerun.io/viewer

### Command Line
- Navigate to assets at this link https://github.com/rerun-io/rerun/releases/
- Download the executable `rerun-cli-0.22.1-x86_64-pc-windows-msvc.exe` or similar
- Rename executable to `rerun.exe`
- Add path of folder or exe to windows environment path

