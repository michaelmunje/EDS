sphinx-apidoc -f -o source/ ../eds/ --separate
sphinx-build -b html .\source\ .\build\
