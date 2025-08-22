CC=mpicc
CFLAGS=-O3 -march=native -lm -std=c99
OBJ=montecarlo

all:
        $(CC) $(OBJ).c -o $(OBJ) $(CFLAGS)

clean:
        rm $(OBJ)
