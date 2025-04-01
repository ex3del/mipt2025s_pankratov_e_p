Как запустить контейнер для тетрадок
cd .\build\notebook
docker build -t my-jupyter .
docker run --gpus all -p 8888:8888 -v ${PWD}:/app my-jupyter (WINDOWS)
docker run --gpus all -p 8888:8888 -v $(pwd):/app my-jupyter (MAC, LINUX)
После запуска в консоли появится ссылка вида: http://127.0.0.1:8888/lab?token=...
Откройте её в браузере.
