#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <iostream>
#include <string>
#include <sstream>
#include "sparsematrix.h"
#include "jacobi_par.h"

template <typename T>
std::string Str( const T & t ) {
   std::ostringstream os;
   os << t;
   return os.str();
}

class Preconditioner {
  protected:
    int iterations;
    std::string name;
    std::string confDesc = "";

  public:

  	virtual void solveLower(SparseMatrix *I, double *b, double *x) = 0;
  	virtual void solveUpper(SparseMatrix *I, double *b, double *x) = 0;
  	virtual void config() = 0;

  	void setIterations(int iter) {
  		iterations = iter;
  	}

  	std::string getLogDesc() {
  		return name + ";" + confDesc;
  	}

  	Preconditioner() {
  		name = "abstract base class";
  	}
};

class JacobiDefault: public Preconditioner {
  protected:
    int nBlock;
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
        std::cin >> nBlock;

        std::cout << "choose Jacobi iterations:\n";
        std::cin >> iterations;

        help = new double[nBlock * nBlock] {};

        confDesc += Str(iterations);
  	}

  	JacobiDefault() {
  		name = "def";
  	}

  	~JacobiDefault() {
  		delete[] help;
  	}
};

class JacobiConcurrent: public Preconditioner {
  protected:

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

        confDesc += Str(iterations);
  	}

  	JacobiConcurrent() {
  		name = "con";
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
		confDesc += Str(secNum) + std::string(";");  
        // i think i need to do something to delete arrays if this is not the first configuration.
        // but even if there is a memory leak, it's so small, it wont be a problem.

        // init block configrations
        secConf = new int[secNum];
        secIter = new int[secNum];
        for (int i = 0; i < secNum; i++) {
              
          std::cout << "choose Async Block count for section: " << i << "\n";
          std::cin >> secConf[i];

          if (i == 0) {
          	confDesc += Str(secConf[i]);
          } else {
          	confDesc += std::string(",") + Str(secConf[i]);
          }
        }

        confDesc += ";";

        for (int i = 0; i < secNum; i++) {
  
          std::cout << "choose iteration count for section: " << i << "\n";
          std::cin >> secIter[i];

          if (i == 0) {
          	confDesc += Str(secIter[i]);
          } else {
          	confDesc += std::string(",") + Str(secIter[i]);
          }
        }
  	}

  	JacobiBlockAsync() {
  		name = "bas";
  	}

  	~JacobiBlockAsync() {
  		delete[] secConf;
  		delete[] secIter;
  	}
};


#endif //PRECONDITIONER_H