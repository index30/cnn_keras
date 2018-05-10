# Make file
.PHONY: clean
env:
	chmod +x ./shell/build_env.sh
	./shell/build_env.sh


clean:
	rm -rf bin
	rm -rf lib
	rm -rf include
