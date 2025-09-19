# LinkedIn-Job-Project (Backend)

## TLDR
This file provides a high-level reason and explaination to what each file and folder does as well as it's location.

## Folders & Files

### Server.py
Where the backend server apis are located, this will communicate with the frontend and run all the `*-algo.py` files. 

### datasets
Where the `LinkedIn2023-2024 data` is located. Though all 31 columns of the data are split across `12 different csv files`.

### *-Task Folders
Holds the following files.

#### *-algo.py
Specific algorithm implementation required for the given task.

#### pre-processing.py
Specific pre-processing implementation required for the given task.

#### *-args.yml
Arguement input file to test inputs for the pre-processing.py and *-algo.py files.

#### tester.ipynb
Test file to run various input patterns into pre-processing.py and *-algo.py and print charts and important metrics for the given task.
