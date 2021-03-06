#!/bin/bash

# delete existent files
rm Julia.gitignore MATLAB.gitignore .gitignore

# create new empty .ignore
touch .gitignore

# julia
wget https://raw.githubusercontent.com/github/gitignore/master/Julia.gitignore

# matlab - octave
wget https://raw.githubusercontent.com/github/gitignore/499ae899e7b54e701e878759f73d9092302fd07a/Global/MATLAB.gitignore

cat Julia.gitignore MATLAB.gitignore OSsFiles.gitignore Custom.gitignore > .gitignore

exit
