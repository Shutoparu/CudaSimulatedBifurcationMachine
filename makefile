default: lib/sbm.so
	python3 main.py

lib/sbm.so: sbm.c
	nvcc --compiler-options -fPIC -shared sbm.c -o lib/sbm.so