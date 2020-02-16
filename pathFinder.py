#wird nur bei den Funktionen saveNetwork und loadNetwork benötigt
#enthält:
#   -getSlash (wichtig z.B. bei verschiedenen Betriebssystemen -> / oder \ )
#   -getCurrentDir
#   -getFilePath


from platform import system
from os.path import join, dirname, realpath

def getSlash():
    
    os=system()
    if os=="Windows":
        slash="\\"
    elif os=="Linux":
        slash="/"

    return slash

def getCurrentDir():
    dirPath = dirname(realpath(__file__))

    return dirPath

def getFilePath(directoryNames,filename,fromCurrentDir=True):
    """
    example:
    directoryNames=["dir1","dir2"]
    filename="file1"

    >>> getFilePath(["dir1","dir2"],"file1")
    'dir1\\dir2\\file1'
    """
    if fromCurrentDir:
        currentDir=getCurrentDir()
        filePath= join(currentDir,*directoryNames,filename)
        
    else:
        filePath= join(*directoryNames,filename)

    return filePath

    



