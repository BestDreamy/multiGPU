
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -g -G
LD_FLAGS    = -lcudart -lcuda
EXE	        = fabricMem
OBJ	        = main.o

default: $(EXE)

main.o: main.cu 
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)
	./$(EXE)

clean:
	rm -rf *.o $(EXE)