# Test-Driven Development (TDD) Analysis Tool with Pydriller

## Prerequisites

- Python 3.9.4
- Java 17

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

Before running we the script we need to setup a Py4J gateway server, this is needed for parsing Java files (Up to Java 17), simply run this shell command file, it contains the necessary command to compile the relevant java files and run the gateway:

For MacOS/linux
```
./setup.sh 
```

For Windows:
```
setup.bat
```
Once the server is running, you may proceed to run the scripts:

Debugging run, script with less try/catch blocks for faster fault localization:
```
python drill_v2_debug.py
```

Official run, script with more logging statements for easier inspection:
```
python drill_v2.py
```


## How Does The Script Works

This tool analyzes Git repositories to detect and measure Test-Driven Development practices by examining the relationship between test and source files over the project's history.

### Overview

The tool analyzes Git repositories to:
1. Identify test and source file pairs
2. Determine when tests were written relative to their corresponding source files
3. Evaluate the strength of test-source relationships
4. Measure TDD adoption patterns across the project's history

### Key Assumptions

1. **Test-Source Relationship**: Each test file should correspond to at least one source file. This is used to identify legitimate test-source pairs.

2. **TDD Commit Patterns**: For files committed together:
   - Tests should be comprehensive (not skeletal)
   - Source files should be skeletal (indicating implementation follows tests)
   - Test files must include proper annotations and assertions

3. **Testing Coverage**: Not all source files require corresponding test files, as some may be utility classes, interfaces, or non-business logic code.

4. **Test Validity**: Valid tests must:
   - Include proper test annotations (e.g., `@Test`)
   - Contain meaningful assertions
   - Not be abstract test classes

### Methodology

#### 1. File Classification

Analyzes each commit in the default branch to identify:

##### Test Files
- Identified through:
  - File naming patterns (e.g., `*Test.java`, `Test*.java`)
  - Content analysis (`@Test` annotations)
  - Framework imports (JUnit, TestNG, etc.)
- Excludes abstract test classes

##### Source Files
- Identified through:
  - File extensions (`.java`, `.py`)
  - Content analysis
  - Relationship to test files

#### 2. File History Analysis

For each unique file, creates:

- **Content Object** tracking:
  - Methods
  - Frameworks used
  - Class names
  - Import statements
  - Dependencies

- **History Object** tracking:
  - Creation date
  - Modification dates
  - Movement history
  - Related files
  - Commit information

#### 3. Test-Source Matching

Calculates a confidence score for each potential test-source pair based on:

1. **Directory Analysis**
   - Same directory location
   - Related directory structures (e.g., `src/test` → `src/main`)

2. **Name Correlation**
   - Direct matches (e.g., `UserTest.java` → `User.java`)
   - Pattern matching for various naming conventions

3. **Content Analysis**
   - Source class references in test file
   - Matching method names
   - Import statements
   - Framework usage

4. **Historical Analysis**
   - Creation timing
   - Modification patterns
   - Repository movement history

#### 4. Secondary Matching

For unmatched files:

##### Test Files
- Analyzes import statements
- Examines method references
- Reviews associated commits

##### Source Files
- Filters non-business logic files
- Identifies utility classes
- Categorizes interfaces and templates

#### 5. TDD Metrics Calculation

Calculates TDD adoption metrics based on:

1. **Temporal Analysis**
   - Test-first commits (tests created before source)
   - Test-after commits (tests created after source)
   - Simultaneous commits (test and source together)

2. **TDD Pattern Detection**
For simultaneous commits:
   - Tests must be comprehensive (>50 lines, contains assertions) (**Pending Justification**)
   - Source must be skeletal (minimal implementation)
   - Framework usage must be proper

### Output

The tool generates detailed reports including:

- Test-source pair matches with confidence scores
- TDD adoption metrics over time
- Directory relationship analysis
- Framework usage statistics
- Code movement patterns

### Notes

- The tool assumes standard project structures but can handle variations
- Confidence scores are relative and should be reviewed
- Historical analysis may be affected by repository reorganizations
- Some matches may require manual verification