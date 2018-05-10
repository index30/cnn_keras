# Make file
.PHONY: all
all: env

.PHONY: env
env:
	chmod +x ./shell/build_env.sh
	./shell/build_env.sh

.PHONY: clean
clean:
	rm -rf bin
	rm -rf lib
	rm -rf include
