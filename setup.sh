#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p java/build

# Compile Java files
javac -cp "java/lib/*" -d java/build java/src/*.java

# Run the Java parser (optional)
java -cp "java/build:java/lib/*" JavaParser
