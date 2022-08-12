IMAGE_NAME=iresnet
CONTAINER_NAME=iresnet
PORT=8097
SHM_SIZE=2g
FORCE_RM=true

build:
	docker build \
		-f Dockerfile \
		-t $(IMAGE_NAME) \
			--no-cache \
		--force-rm=$(FORCE_RM) \
		.
restart: stop start

start:
	docker run \
		-dit \
		-v $(PWD):/workspace \
		-p $(PORT):$(PORT) \
		--name $(CONTAINER_NAME) \
		--rm \
		--shm-size $(SHM_SIZE) \
		$(IMAGE_NAME)

stop:
	docker stop $(IMAGE_NAME)

attach:
	docker exec \
		-it \
		$(CONTAINER_NAME) bash 