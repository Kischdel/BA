CC=g++
MKLROOT=/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/
CFLAGS=-fopenmp -m64 -I${MKLROOT}include  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl


SOURCE=./src/
TARGET=./build/

$(TARGET)prog: $(TARGET)main.o $(TARGET)jacobi_par.o $(TARGET)parex.o $(TARGET)fgmres.o $(TARGET)sparsematrix.o
	$(CC) $(CFLAGS) $(CFLAGS3) -o $(TARGET)prog $(TARGET)main.o $(TARGET)jacobi_par.o $(TARGET)parex.o $(TARGET)fgmres.o $(TARGET)sparsematrix.o

$(TARGET)main.o: $(SOURCE)main.cpp
	$(CC) $(CFLAGS) -c -o $(TARGET)main.o $(SOURCE)main.cpp	

$(TARGET)jacobi_par.o: $(SOURCE)jacobi_par.cpp
	$(CC) $(CFLAGS) -c -o $(TARGET)jacobi_par.o $(SOURCE)jacobi_par.cpp
 
$(TARGET)fgmres.o: $(SOURCE)fgmres.cpp
	$(CC) $(CFLAGS) -c -o $(TARGET)fgmres.o $(SOURCE)fgmres.cpp
 
$(TARGET)parex.o: $(SOURCE)parex.cpp
	$(CC) $(CFLAGS) -c -o $(TARGET)parex.o $(SOURCE)parex.cpp

$(TARGET)sparsematrix.o: $(SOURCE)sparsematrix.cpp
	$(CC) $(CFLAGS) -c -o $(TARGET)sparsematrix.o $(SOURCE)sparsematrix.cpp