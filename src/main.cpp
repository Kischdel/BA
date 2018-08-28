#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include "blas.h"
#include "parex.h"
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

  double h_pow_2 = 1.0 / (nBlock + 1);
  h_pow_2 = h_pow_2 * h_pow_2;
  int counter = 0;
  
  for(int k = 1; k <= nBlock; k++) {
    for(int m = 1; m <= nBlock; m++ ) {
      
      double x = (double)m / (double)(nBlock + 1);
      double y = (double)k / (double)(nBlock + 1);
      
      b[counter++] = h_pow_2 * (32 * ( x * (1.0 - x) + y * (1.0 - y)));
    }
  }
}


int main(int argc, char *argv[]) {
  
  SparseMatrix *A;
  double tol = 1.0e-09;

  bool running = true;

  // simple menu
  while(running) {
    std::cout << "choose action:\n";
    std::cout << "1) compute matrix\n";
    std::cout << "2) load Matrix\n";
    std::cout << "3) solve single FGMRES\n";
    std::cout << "4) map out steps/runtime of FGMRES\n";
    std::cout << "5) execute jacobi\n";
    std::cout << "6) not defined test section\n";
    std::cout << "7) quit\n";
  
    // read keyboard input
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

      // solve the computed/loaded matrix with FGMRES
      case 3:
        {
          int nBlock = A->nBlock;
          int n = A->n;
          
          // initialize right side
          double *b = new double[n];
          calculateB(nBlock, b);

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

          // run FGMRES
          resultFGMRES res;
          epsilonfem::CSR_Solver *solver = new epsilonfem::CSR_Solver();
          solver->FGMRES(A, b, x, tol, restartFGMRES, r, iterationJacobi, &res);

          printVector("result", n, r);

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
    
          // initialize right side
          double *b = new double[n];
          calculateB(nBlock, b);
  

          // setup logging
          std::ostringstream buildfilename;
          buildfilename << "./log/mapping_Blocksize_";
          buildfilename << nBlock;
          buildfilename << ".txt";
          std::string filename = buildfilename.str(); 
  
          std::ofstream myfile;
          myfile.open(filename.c_str(), std::ios_base::app);
  
          if (myfile.is_open()) {
            
            // define restart range            
            int restartStart = 250;
            int restartEnd = 550;
            int restartStep = 25;

            // define jacobi iteration range
            int iterStart = 4;
            int iterEnd = 10;
            int iterStep = 2;

            for(int k = iterStart; k <= iterEnd; k += iterStep) {
              for(int i = restartStart; i <= restartEnd; i += restartStep) {

                // initialize resultStruct for logging

                resultFGMRES res;
                res.restart = i;
                res.iter = k;

                // initialize x0
                double *x = new double[n] {};
  
                // initialize result
                double *r = new double[n] {};


                // run FGMRES
                epsilonfem::CSR_Solver *solver = new epsilonfem::CSR_Solver();
                solver->FGMRES(A, b, x, tol, i, r, k, &res);


                // write to logfile
                myfile << res.restart;
                myfile << ";";
                myfile << res.iter;
                myfile << ";";
                myfile << res.steps;
                myfile << ";";
                myfile << res.restarts;
                myfile << ";";
                myfile << res.time;
                myfile << ";";
                myfile << res.timeFgmres;
                myfile << ";";
                myfile << res.averageTimeFgmres;
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

      // execute jacobi only  
      case 5:
        {
          int nBlock = A->nBlock;
          int n = A->n;
          
          // initialize iterations
          int iterations;
          std::cout << "choose jacobi iterations:\n";
          std::cin >> iterations;

          // compute right side
          double *b = new double[n] {};
          calculateB(nBlock, b);
    
          // initialize x0, y, r and s
          double *x = new double[n] {}; // initial guess
          double *y = new double[n] {}; // temp for residual and jacobi version without concurrent access
          double *r = new double[n] {}; // rigth side like in first FGMRES iteration
          double *s = new double[n] {}; // store result between lower and upper 

          // compute vector from first FGMRES iteration
          /* i dont know if this is a good test
          const char transpose = 'n';
          epsilonfem::EFEM_BLAS::dcsrgemv(&transpose, &n, A->val, A->row, A->col, x, r);
          epsilonfem::EFEM_BLAS::daxpy (n, -1, b, 1, r, 1);
          double g = epsilonfem::EFEM_BLAS::dnrm2 (n, r, 1);
          epsilonfem::EFEM_BLAS::dscal (n, 1.0/g, r, 1);
          */

          //pointer to upper and lower
          void (*lower)(const int, const int, const int, double*, double*, int*, int*, double*) = &lowerLine;
          void (*upper)(const int, const int, const int, double*, double*, int*, int*, double*) = &upperLine;
          
          // experimental jacobi version with teaming and overhead
          //jacobi(A, iterations, b, x, lower);
          //jacobi(A, iterations, x, y, upper);

          // start messurement
          std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

          // fully syncronized jacobi version
          jacobiLowerSync(A, iterations, b, s, y);
          std::chrono::high_resolution_clock::time_point middle = std::chrono::high_resolution_clock::now();

          for(int i = 0; i < n; i++) {
            x[i] = 0;
            y[i] = 0;
          }

          // simpleJacobiUpper(A, iterations, s, x, y);
          
          jacobiUpperSync(A, iterations, s, x, y);
          //simpleJacobiUpper(A, iterations, s, x, y);
          //jacobiUpperHalfSync(A, iterations, s, x);

          //simpleJacobi(A, iterations, b, x, y);

          //printVector("s", n, s);
          //printVector("x", n, x);

          // fully syncronized jacobi version with concurrent access
          //jacobiLowerHalfSync(A, iterations, b, s);
          //std::chrono::high_resolution_clock::time_point middle = std::chrono::high_resolution_clock::now();
          //jacobiUpperHalfSync(A, iterations, s, x);


          // end messurement
          std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

          auto executionTimeLower = std::chrono::duration_cast<std::chrono::microseconds>(middle - start).count();
          auto executionTimeUpper = std::chrono::duration_cast<std::chrono::microseconds>(end - middle).count();
  
          double residualNormLower = computeResidualLower(A, b, s, y);
          double residualNormUpper = computeResidualUpper(A, s, x, y);

          
          std::cout << "iter: " << iterations << " / residualNorm: ";
          std::cout << residualNormLower << " / ";
          std::cout << residualNormUpper << " time: ";
          std::cout << (double)executionTimeLower / 1000000 << " / ";
          std::cout << (double)executionTimeUpper / 1000000 << "\n";

          delete[] b;
          delete[] x;
          delete[] y;
          delete[] r;
          delete[] s;
        }
        break;

      //experiments
      case 6:
        {



          /* matrix multiplication
          for(int i = 0; i < A->n; i++) {
            for(int j = 0; j < A->n; j++) {

              double val = 0;
              
              for(int k = 0; k < A->n; k++) {

                val += A->getLowerILUValAt(i,k) * A->getUpperILUValAt(k,j);

              }
              std::cout << val << "\t";
            }
            std::cout << "\n";
          }
          */
        }
        break;

      // quit the program
      case 7:
        running = false;
        break;

      default:
        running = false;
    }
  }

  return 0;
}