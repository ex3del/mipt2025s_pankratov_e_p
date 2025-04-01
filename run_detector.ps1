# run_detector.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
docker-compose up --build -d
docker-compose exec app bash