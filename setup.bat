@echo off
mkdir java\build 2>nul

REM Compile Java files
javac -cp "java\lib\*" -d "java\build" "java\src\*.java"

REM Optional: Run the parser
java -cp "java\build;java\lib\*" JavaParser

pause