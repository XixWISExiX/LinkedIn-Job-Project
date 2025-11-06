# How to run the application
```bash
streamlit run dashboard.py 
```

## How to download the data
1) Click the following link: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

2) Click the download button

3) Then Click download dataset as .zip

4) Take `archive.zip` and place it into the following directory location `datasets`

5) Extract the archive file and the data should be ready for the application to use.

## Folders & Files
A high-level reason and explaination to what each file and folder does as well as it's location.

### datasets
Where the `LinkedIn2023-2024 data` is located. Though all 31 columns of the data are split across `12 different csv files`.

### *_task Folders
Holds the following files.

#### *_algo.py
Specific algorithm implementation required for the given task.

#### pre_processing.py
Specific preprocessing implementation required for the given task.

#### *_args.yml
Arguement input file to test inputs for the pre\_processing.py and *\_algo.py files.
