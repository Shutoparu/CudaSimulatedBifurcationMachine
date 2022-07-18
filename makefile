main: lib/sbm_cu.so
	python3 main.py

main_c: lib/sbm_c.so
	python3 main_c.py

lib/sbm_cu.so: sbm.cu
	nvcc --compiler-options -fPIC -shared sbm.cu -o lib/sbm_cu.so

lib/sbm_c.so: sbm.c
	nvcc --compiler-options -fPIC -shared sbm.c -o lib/sbm_c.so

debug: coreDump
	cuda-gdb ./coreDump
	rm coreDump
coreDump: sbm.cu
	nvcc -G -g -o coreDump sbm.cu

c: bin/sbm_cu.o
	./bin/sbm_cu.o

bin/sbm_cu.o: sbm.cu
	nvcc -o bin/sbm_cu.o sbm.cu

runData: lib/sbm_cu.so
	python3 toQUBO.py
