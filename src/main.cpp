#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "mkl.h"
#include "parex.h"
#include "fgmres.h"
#include "sparsematrix.h"
#include "jacobi_par.h"
#include "CSR_Solver.h"

void printVector(std::string s, int n, const double *v) {
  std::cout << s << "\n";
  for(int i = 0; i < n; i++) {
    std::cout << v[i] << "\n";
  }
  std::cout << "\n";
}

int main(int argc, char *argv[]) {
  
  SparseMatrix *A;
  bool running = true;

  // simple menu
  while(running) {
  	std::cout << "choose action:\n";
		std::cout << "1) compute Matrix\n";
		std::cout << "2) load Matrix\n";
		std::cout << "3) solve\n";
		std::cout << "4) quit\n";
	

		int input;
		std::cin >> input;

		switch(input) {
			case 1:
				{
					std::cout << "choose blocksize:\n";
					std::cin >> input;

					A = new SparseMatrix(input);
					A->computeILU();
					A->saveToFile();
				}
				break;

			case 2:
				{
					std::string filename = "./res/m_Blocksize_256.txt";
  				A = new SparseMatrix(filename);
  			}
				break;

			case 3:
				{
					int nBlock = A->nBlock;
  				int n = A->n;

				  double tol = 1.0e-03;
	  
			  	// compute b
 			 		double b[n];
  				int counter = 0;
  
  				for(int k = 1; k <= nBlock; k++) {
    				for(int m = 1; m <= nBlock; m++ ) {
      
      				double x = (double)m / (double)(nBlock + 1);
    	  			double y = (double)k / (double)(nBlock + 1);
      
  	    			b[counter++] = 32 * ( x * (1.0 - x) + y * (1.0 - y));
	    			}
  				}
  
  				// initialize x0
  				double x[n];
  				for(int i = 0; i < n; i++) {
	    			x[i] = 0;
  				}
  
  				// initialize result
  				double r[n];
  				for(int i = 0; i < n; i++) {
    				r[i] = 0;
	  			}

  				// FGMRES
  				epsilonfem::CSR_Solver *solver = new epsilonfem::CSR_Solver();
  				solver->FGMRES(A, b, x, tol, 20, r, 1);
  			}
				break;

			case 4:
				running = false;
				break;

			default:
				running = false;
		}
  }

  return 0;
}