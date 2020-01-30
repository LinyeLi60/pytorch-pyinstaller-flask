# pytorch-pyinstaller-flask
deploy pytorch model using flask, and use pyinstaller to freeze pytorch and flask into standalone executable file.

## 运行环境
Python 3


## 第三方库
- [flask]
- [pytorch]


## 环境配置
``` Python
pip install flask==1.1.1
pip install torch==1.3.1
```


## 使用帮助
``` 
Command line mode:
> python server.py
> python client.py

Executable mode:
> pyinstaller -F server.py --name=server
> server.exe
> python client.py
```
