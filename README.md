# Embedded perception stack

## How to run:
```
cmake --build .
```

## How to setup vscode

* tasks.json:
```
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "cmake",
            "type": "shell",
            "options": {
                "cwd": "${workspaceRoot}/build"
            },
            "command": "cmake --build  ${workspaceRoot}/build/",
            "problemMatcher": [],
            "group": {
            "kind": "build",
            "isDefault": true
            }
        }
    ]
}
```

* launch.json:
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "debug main",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceRoot}/build/emb_perception",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${fileDirname}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                },
                {
                    "description": "Set Disassembly Flavor to Intel",
                    "text": "-gdb-set disassembly-flavor intel",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "C/C++: cpp build active file",
            "miDebuggerPath": "/usr/bin/gdb"
        }
    ]
}
```

* Note: Generate `compile_commands.json` and `c_cpp_properties.json`

## Not included:

* libtorch (~1G) (has to be downloaded from pytorch.org)
* local IDE setup