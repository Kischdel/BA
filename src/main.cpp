#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "mkl.h"
#include "parex.h"
#include "fgmres.h"
#include "sparsematrix.h"
#include "jacobi_par.h"
#include "CSR_Solver.h"
#include "result.h"

void printVector(std::string s, int n, const double *v) {
  std::cout << s << "\n";
  for(int i = 0; i < n; i++) {
    std::cout << v[i] << "\n";
  }
  std::cout << "\n";
}

// compute right side for a loaded matrix
void calculateB(int nBlock, double* b) {

  int counter = 0;
  
  for(int k = 1; k <= nBlock; k++) {
    for(int m = 1; m <= nBlock; m++ ) {
      
      double x = (double)m / (double)(nBlock + 1);
      double y = (double)k / (double)(nBlock + 1);
      
      b[counter++] = 32 * ( x * (1.0 - x) + y * (1.0 - y));
    }
  }
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
    std::cout << "4) map out steps/runtime\n";
		std::cout << "5) quit\n";
	

		int input;
		std::cin >> input;

		switch(input) {
      // compute and store a matrix
			case 1:
				{
					std::cout << "choose blocksize:\n";
					std::cin >> input;

					A = new SparseMatrix(input);
					A->computeILU();
					A->saveToFile();
				}
				break;

      // load a previous computed matrix
			case 2:
				{
          std::cout << "choose blocksize:\n";
					std::cin >> input;
                   
          std::ostringstream buildfilename;
          buildfilename << "./res/m_Blocksize_";
          buildfilename << input;
          buildfilename << ".txt";
          std::string filename = buildfilename.str();
           
  				A = new SparseMatrix(filename);
  			}
				break;

      // solve the computetd/loaded matrix with FGMRES
			case 3:
				{
					int nBlock = A->nBlock;
  				int n = A->n;

				  double tol = 1.0e-09;
	  
			  	// compute b
 			 		//double b[n];
          //double *b = (double*)malloc(n * sizeof(double));
          double *b = new double[n];
  				calculateB(nBlock, b);
          
          printVector("b: ", n, b);

  				// initialize x0
  				double *x = new double[n] {};

  				// initialize result
  				double *r = new double[n] {};

          // specify parameters
          int restartFGMRES;
          int iterationJacobi;

          std::cout << "choose FGMRES restart size:\n";
          std::cin >> restartFGMRES;

          std::cout << "choose jacobi iterations:\n";
          std::cin >> iterationJacobi;

  				// FGMRES
          resultFGMRES res;
  				epsilonfem::CSR_Solver *solver = new epsilonfem::CSR_Solver();
  				solver->FGMRES(A, b, x, tol, restartFGMRES, r, iterationJacobi, &res);

          //delete
          delete[] b;
          delete[] x;
          delete[] r;
  			}
				break;

      //map out runntime and steps for a loaded matrix
      case 4:
        {
          int nBlock = A->nBlock;
          int n = A->n;

          double tol = 1.0e-09;
    
          // compute b
          double *b = new double[n];
          calculateB(nBlock, b);
  

          // setup logging
          std::ostringstream buildfilename;
          buildfilename << "./log/mapping_Blocksize_";
          buildfilename << nBlock;
          buildfilename << ".txt";
          std::string filename = buildfilename.str(); 
  
          std::ofstream myfile;
          myfile.open(filename.c_str());
  
          if (myfile.is_open()) {
            
            // define restart range            
            int restartStart = 140;
            int restartEnd = 300;
            int restartStep = 10;

            // define iteration range
            int iterStart = 4;
            int iterEnd = 20;
            int iterStep = 2;

            for(int k = iterStart; k <= iterEnd; k += iterStep) {
              for(int i = restartStart; i <= restartEnd; i += restartStep) {

                // initialize resultStruct

                resultFGMRES res;
                res.restart = i;
                res.iter = k;

                // initialize x0
                double *x = new double[n] {};
  
                // initialize result
                double *r = new double[n] {};


                // FGMRES
                epsilonfem::CSR_Solver *solver = new epsilonfem::CSR_Solver();
                solver->FGMRES(A, b, x, tol, i, r, k, &res);


                // write to logfile
                myfile << res.restart;
                myfile << ";";
                myfile << res.iter;
                myfile << ";";
                myfile << res.steps;
                myfile << ";";
                myfile << res.time;
                myfile << ";";
                myfile << res.timeJacobi;
                myfile << ";";
                myfile << res.averageTimeJacobi;
                myfile << ";";

                myfile << "\n";
                myfile.flush();

                delete[] x;
                delete[] r;
              }

            }      
          } else {
            std::cout << "file Error" << "\n";
          }

          // end logging
          myfile.close();

          delete[] b;
        }
        break;

      // quit the program
			case 5:
				running = false;
				break;

			default:
				// just for experiments
        int nBlock = A->nBlock;
        int n = A->n;

        double tol = 1.0e-09;
    
        // compute b
        double b[n];
        calculateB(nBlock, b);
  
        // initialize x0 and y[0]
        double x[n] = {};
        double y[n] = {};

        //pointer to upper and lower
        void (*lower)(const int, const int, const int, double*, double*, int*, int*, double*) = &lowerLine;
        void (*upper)(const int, const int, const int, double*, double*, int*, int*, double*) = &upperLine;

        jacobi(A, 30, b, x, lower);
        jacobi(A, 30, x, y, upper);

        printVector("result: ", n, y);
		}
  }

  return 0;
}