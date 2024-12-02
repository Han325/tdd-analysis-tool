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
Once the server is running, you may proceed to run the scripts **on another terminal**:

Debugging run, script with less try/catch blocks for faster fault localization:
```
python drill_v2_debug.py
```

Official run, script with more logging statements for easier inspection:
```
python drill_v2.py
```


## How Does The Script Works

This tool analyzes Apache Git repositories to investigate Test-Driven Development (TDD) adoption patterns by examining commit histories and relationships between test and source files. The analysis specifically focuses on answering key research questions about TDD practices in Apache projects.

### Research Questions Addressed

1. How does a TDD-compliant commit manifest in practice?
2. What is the temporal relationship between test and source file creation?
   - Test created before source (test-first)
   - Test created after source (test-after)
   - Test and source created in same commit
3. How does commit size impact TDD patterns?
4. What methods can reliably identify test files?
5. How can we establish links between test and source files?
6. What characteristics identify test-first commits?
7. How can we measure TDD adoption rates?
8. How does TDD adoption vary across Apache projects?

### Key Assumptions

1. **Test-Source Relationship**: 
   - Each test file should correspond to at least one source file
   - The git repository should reflect TDD practices through commit history

2. **TDD Commit Patterns**:
   - Tests should be comprehensive (not skeletal)
   - Source files should be skeletal in test-first commits
   - Test files must include proper annotations and assertions
   - Commit messages and content should show TDD intent

3. **Test Validity**: Valid tests must:
   - Include proper test annotations (e.g., `@Test`)
   - Contain meaningful assertions
   - Not be abstract test classes
   - Have proper test framework imports

### Methodology

#### 1. Repository Analysis

Focuses exclusively on Apache projects due to:
- Consistent Apache License coverage
- High-quality project standards
- Best practice adherence
- Ethical considerations alignment

#### 2. File Classification

Analyzes each commit to identify:

##### Test Files
- Identified through:
  - File naming patterns (e.g., `*Test.java`, `Test*.java`)
  - Content analysis (`@Test` annotations)
  - Framework imports (JUnit, TestNG, etc.)
- Excludes abstract test classes

##### Source Files
- Identified through:
  - Java source files
  - Content analysis
  - Relationship to test files

#### 3. Commit Graph Analysis

The tool maintains a directed graph of commits to handle complex repository histories and improve TDD pattern detection accuracy. This graph structure serves three key purposes:

##### Repository Move Detection
- Identifies cases where files were moved from another repository by:
  - Finding files with similar content but different creation dates
  - Checking if commits occur at branch merge points
  - Adjusting confidence scores when repository moves are detected
  - Preventing false test-after classifications due to repository migrations

##### Ancestry Analysis
- Tracks related changes across branches through common ancestor detection:
  - Finds lowest common ancestor between commit pairs
  - Maps evolutionary relationships between test and source files
  - Analyzes modifications across different branches
  - Helps establish true chronological relationships between test and source creation

##### Branch Point Detection
- Identifies critical points in repository structure:
  - Locates commits with multiple parents (merge commits)
  - Marks points where code paths diverged
  - Finds potential migration or integration points
  - Helps adjust timestamp-based analysis when branch operations affect file history

This enhanced understanding of repository structure allows the tool to:
- Account for complex version control operations
- Provide context beyond simple file timestamps
- Make more accurate determinations of test-first vs test-after patterns
- Handle cases where simple chronological analysis would be misleading

#### 4. Test-Source Matching

Calculates confidence scores based on:

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

#### 5. Commit Analysis

Analyzes commit patterns through:

1. **Size Categories**
   - Small (≤2 files, ≤50 lines)
   - Medium (≤5 files, ≤200 lines)
   - Large (>5 files or >200 lines)

2. **Message Analysis**
   - TDD indicators in commit messages
   - Test-related terminology
   - Implementation descriptions

3. **Content Analysis**
   - Test file changes
   - Test framework modifications
   - Implementation patterns

### Output Analysis

The tool generates detailed reports including:

1. **TDD Adoption Metrics**
   - Test-first vs test-after ratios
   - Same-commit test-source pairs
   - TDD pattern compliance rates

2. **Commit Analysis**
   - Size distribution of TDD commits
   - Message pattern analysis
   - Context-based TDD indicators

3. **Project Comparison**
   - Cross-project TDD adoption rates
   - Framework usage patterns
   - Directory structure analysis

### Notes

- Results require careful interpretation due to:
  - Repository restructuring effects
  - Commit squashing impacts
  - Branch merge complications
  - Historical data limitations