NVCC = nvcc
ARCH ?= native

.PHONY: all clean

all: tc_numeric_test.out

tc_numeric_test.out: tc_numeric_test.cu
	nvcc -O3 -arch=native -o $@ $<

clean:
	rm -f tc_numeric_test.out
