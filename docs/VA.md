Small patch for the VA package to run jqscore instead of Chimera+MapQ:

```
https://gist.github.com/aozalevsky/a68db4e0e7161a06435689632b19e6aa
```

Example patch application:

```
# Patch VA to use jqscore
wget https://gist.githubusercontent.com/aozalevsky/a68db4e0e7161a06435689632b19e6aa/raw/ba44a6aea856377097ebab3e766ad2c15a58478a/va_jqscore.patch -O /va_jqscore.patch
patch -d  $(dirname $(python -c "import va; print(va.__file__)")) -p2 < /va_jqscore.patch && rm -rf /va_jqscore.patch
```
