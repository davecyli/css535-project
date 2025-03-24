# css535-project
Repo for the CSS535 Optical Flow CUDA project

## Pre-requisities
- Visual Studio 17 2022
- Windows 11
- CUDA 12.8.0
- Rerun-cli >=0.22.1 
- Pixi >=0.41.4


## OpenCV
Opencv is a critical library that allows us to load video files, manipulate frames and display results.
### Setting up OpenCV
Install the windows x64 version of OpenCV 4.11.0+ to your C: drive. In your C: drive there should be an opencv folder containing the following:
```
build\
sources\
LICENSE.txt
LICENSE_FFMPEG.txt
README.md.txt
```

## Rerun
Rerun is a multimodal data logger and visualizer. This will accelerate the way we observe and analyze our data

### Setting up Rerun
#### Command Line
- Navigate to assets at this link https://github.com/rerun-io/rerun/releases/
- Download the executable `rerun-cli-0.22.1-x86_64-pc-windows-msvc.exe` or similar to a folder such as `C:\\bin`
- Rename executable to `rerun.exe`
- Add path of folder or exe to windows environment path for example `C:\\bin`

## Pixi package management
- Pixi is a package management tool for developers and allows developers to install libraries and applications in a reproducible way.

### Installing Pixi
Run this script in a command prompt
```
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

## Building the Visual Studio Solution
Git clone the source code from https://github.com/davecyli/css535-project
Within a Visual Studio developers CMD and from the project root directory (ex: css535-project/) run:
```
pixi run start
```
This will configure, build and execute from within a reproducible environment. At the end of the script a test executable will run to test connection to your rerun installation and ensure it is in your windows environment path variables.

## Visual Studio Solution
The OpticalFlow.sln file is in the .build folder: .\css535-project\\.build

Three main projects:
- driverCPU - Contains the CPU implementation
- driverGPU - Contains the GPU Naive implementation
- driverGPUSharedMem - Contains the GPU Optimized implementation

### Running the Implementations
Set the mode to Release then build and run in Visual Studio.
When running the projects with no command line arguments provided, the application will request a path to a video file.
Please enter a path without quotes to specify a video or enter nothing which will use the default Squishies.mp4 located in the relative path

# Miscellaneous

## To test profiler
- Open a `Visual Studio 2022 Developer Command Prompt`
- Navigate to ..\css535-project\OpticalFlow

## To build and run development drivers
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

## Verifying GPU Least Squares Solver implementation
Run the program as described above then run:
```
FC expected_results.txt .build\results.txt
```
From within the project root directory to verify that the results you are getting are the same as expected.

