#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <iostream>
#include "sparsematrix.h"
#include "jacobi_par.h"

class Preconditioner {
  protected:
    

  public:

  	virtual void solveLower(SparseMatrix *I, double *b, double *x) = 0;
  	virtual void solveUpper(SparseMatrix *I, double *b, double *x) = 0;
  	virtual void config() = 0;
};

class JacobiDefault: public Preconditioner {
  protected:
    int iterations;
    int dim;
    double *help;

  public:
  	void solveLower(SparseMatrix *I, double *b, double *x) {
  		jacobiLowerSync(I, iterations, b, x, help);
  	}

  	void solveUpper(SparseMatrix *I, double *b, double *x) {
  		jacobiUpperSync(I, iterations, b, x, help);
  	}

  	void config() {

  		// read parameters from keyboard
        std::cout << "what is the Matrix Block size?\n";
        std::cin >> dim;

        std::cout << "choose Jacobi iterations:\n";
        std::cin >> dim;

        help = new double[dim] {};
  	}

  	~JacobiDefault() {
  		delete[] help;
  	}
};

class JacobiConcurrent: public Preconditioner {
  protected:
    int iterations;

  public:
  	void solveLower(SparseMatrix *I, double *b, double *x) {
  		jacobiLowerHalfSync(I, iterations, b, x);
  	}

  	void solveUpper(SparseMatrix *I, double *b, double *x) {
  		jacobiUpperHalfSync(I, iterations, b, x);
  	}

  	void config() {

  		// read parameters from keyboard
        std::cout << "choose Jacobi iterations:\n";
        std::cin >> iterations;
  	}
};

class JacobiBlockAsync: public Preconditioner {
  protected:
    int secNum;
    int *secConf;
    int *secIter;

  public:
  	void solveLower(SparseMatrix *I, double *b, double *x) {
  		jacobi(I, b, x, &lowerLine, secNum, secConf, secIter);
  	}

  	void solveUpper(SparseMatrix *I, double *b, double *x) {
  		jacobi(I, b, x, &upperLine, secNum, secConf, secIter);
  	}

  	void config() {
  		// init section count
        std::cout << "choose jacobi sections:\n";
        std::cin >> secNum;
  
        // i think i need to do something to delete arrays if this is not the first configuration.
        // but even if there is a memory leak, it's so small, it wont be a problem.

        // init block configrations
        secConf = new int[secNum];
        secIter = new int[secNum];
        for (int i = 0; i < secNum; i++) {
              
          std::cout << "choose Async Block count for section: " << i << "\n";
          std::cin >> secConf[i];
        }

        for (int i = 0; i < secNum; i++) {
  
          std::cout << "choose iteration count for section: " << i << "\n";
          std::cin >> secIter[i];          
        }
  	}

  	~JacobiBlockAsync() {
  		delete[] secConf;
  		delete[] secIter;
  	}
};


#endif //PRECONDITIONER_H