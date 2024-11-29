# Pydriller Trials

## Setup

To run the pydriller script, setup a virtual environment first (This example uses virtualenv, feel free to use any package)

```
virtualenv venv 
```

Activate the existing the virtual environment using this command, this is for virtualenv only (For macOS/linux)

```
. venv/bin/activate  
```


Afterwards pip install the dependencies:

```
pip install -r requirements.txt 
```

Finally run the command:
```
python drill.py
```



- Go through all the commits
- For each commit, we go through each files in the commit, analyze file content, at that point we
  determine whether if its source file or a test file by analyzing the file, for annotations, i.e imports for junit, @Test, and we create comprehensive object that stores every info
- After that we find the matched pair, here we use the criteria of:
    - directory (what is the criteria, in the is_related_directory)
    - commit history (?), 
- Analyze: 
    Calculate TDD Ratio based on:
        (new) calculate ratio between business logic files to test file, i.e having 16 matched pairs should mean that there are 16 business logic files, exact match means is more tdd, lesser probably not. 
        how many test first commit, how many test after commit
        if its in the same commit, check the content (pending justification) of the files, and determine whether if its tdd. 
