main: lib/sbm.so
	python3 main.py

subMain: lib/sbm.so
	python3 subMain.py

lib/sbm.so: sbm.cu
	nvcc --compiler-options -fPIC -shared sbm.cu -o lib/sbm.so