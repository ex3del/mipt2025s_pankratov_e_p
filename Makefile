.PHONY: detector
detector:
	docker-compose up --build -d
	docker-compose exec app bash