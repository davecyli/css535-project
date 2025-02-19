# css535-project
Repo for the CSS535 Optical Flow CUDA project

## Pre-requisities
- VS 2022
- Windows 11
- CUDA 12.8.0

## To test CUDA installation
- Open a `Visual Studio 2022 Developer Command Prompt`
- Navigate to ..\css535-project\OpticalFlow
- Run the following command to test your CUDA installation
```
$ nvcc test.cu
```
- Run output
```
$ .\a.exe
```
- Expected Results:
```
C:\Users\davec\OneDrive\School\CSS 535\project\OpticalFlow>.\a.exe
{1,2,3,4,5} + {10,20,30,40,50} = {11,22,33,44,55}
```

## Notes
Should be able to ignore this warning
```
08:06:11 davec@dli-alienware css535-project ±|develop ✗|→ git add .
warning: in the working copy of 'OpticalFlow/test.cu', LF will be replaced by CRLF the next time Git touches it
```