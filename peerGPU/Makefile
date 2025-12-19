
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -g -G
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = peerGPU
OBJ	        = main.o

default: $(EXE)

main.o: main.cu kernel.cu
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)