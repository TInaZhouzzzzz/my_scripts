objects = testInt8.o testFp32.o testFp16.o
edit: $(objects)
testInt8.o : testGemmInt8.cpp
	/usr/local/cuda-12.1/bin/nvcc -lcublas -std=c++11 -o testInt8.o testGemmInt8.cpp -O3
testFp32.o : testGemmFp32.cpp
	/usr/local/cuda-12.1/bin/nvcc -lcublas -std=c++11 -o testFp32.o testGemmFp32.cpp -O3
testFp16.o : testGemmFp16.cpp
	/usr/local/cuda-12.1/bin/nvcc -lcublas -std=c++11 -o testFp16.o testGemmFp16.cpp -O3
.PHONY:clean
clean: $(objects)
	rm $(objects)
