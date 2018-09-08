#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <omp.h>
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

// write Result from result struct to logfile
void writeResult(std::ofstream *file, resultFGMRES *res) {

	*file << res->restart << ";";
  *file << res->iter << ";";
  *file << res->steps << ";";
  *file << res->restarts << ";";
  *file << res->time << ";";
  *file << res->timeFgmres << ";";
  *file << res->averageTimeFgmres << ";";
  *file << res->timeJacobiLower << ";";
  *file << res->averageTimeJacobiLower << ";";
  *file << res->timeJacobiUpper << ";";
  *file << res->averageTimeJacobiUpper << ";";
  *file << res->averageNormJacobiLower << ";";
  *file << res->averageNormJacobiUpper << ";";
	
  *file << "\n";
  file->flush();
}


int main(int argc, char *argv[]) {
  
  // initilization
  epsilonfem::CSR_Solver *solver = new epsilonfem::CSR_Solver();
  SparseMatrix *A;
  double tol = 1.0e-09;
  bool running = true;

  // simple menu
  while(running) {
  	// write menu
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

    // switch between actions
    switch(input) {
      // compute and store a matrix
      case 1:
        {
          std::cout << "choose blocksize:\n";
          std::cin >> input;

          // construct and save matrix
          A = new SparseMatrix(input);
          A->computeILU();
          A->saveToFile();
        }
        break;

      // load a previous computed matrix
      case 2:
        {
        	// read blocksize from keyboard 
          std::cout << "choose blocksize:\n";
          std::cin >> input;
          
          // build filename
          std::ostringstream buildfilename;
          buildfilename << "./res/m_Blocksize_";
          buildfilename << input;
          buildfilename << ".txt";
          std::string filename = buildfilename.str();
           
          // load specified matrix
          A = new SparseMatrix(filename);
        }
        break;

      // solve the computed/loaded matrix with FGMRES
      case 3:
        {
        	// initialize dimensions
          int nBlock = A->nBlock;
          int n = A->n;
          
          // initialize right side
          double *b = new double[n];
          calculateB(nBlock, b);

          // initialize x0
          double *x = new double[n] {};

          // initialize result
          double *r = new double[n] {};

          // parameters for execution
          int restartFGMRES;
          int iterationJacobi;
          int repetitions;

          // setup logging
          std::ostringstream buildfilename;
          buildfilename << "./log/mapping_Blocksize_";
          buildfilename << nBlock;
          buildfilename << ".txt";
          std::string filename = buildfilename.str(); 

          std::ofstream myfile;
          myfile.open(filename.c_str(), std::ios_base::app);

          if (myfile.is_open()) {

          	// read parameters from keyboard
          	std::cout << "choose FGMRES restart size:\n";
          	std::cin >> restartFGMRES;
	
          	std::cout << "choose jacobi iterations:\n";
          	std::cin >> iterationJacobi;

          	std::cout << "choose how often to repeat the execution:\n";
          	std::cin >> repetitions;

          	for(int i = 0; i < repetitions; i++) {
	
          		// initialize resultStruct for logging
          		resultFGMRES res;
          		res.restart = restartFGMRES;
            	res.iter = iterationJacobi;
	
							// run FGMRES
          		solver->FGMRES(A, b, x, tol, restartFGMRES, r, iterationJacobi, &res);
		
          		writeResult(&myfile, &res);
        		}
          }

          //printVector("result", n, r);

          //free allocated memory
          delete[] b;
          delete[] x;
          delete[] r;
        }
        break;

      //map out runntime and steps for a loaded matrix
      case 4:
        {
        	// initialize dimensions
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
            int restartStart;
            int restartEnd;
            int restartStep;
            std::cout << "choose start of FGMRES restart range:\n";
          	std::cin >> restartStart;
          	std::cout << "choose end of FGMRES restart range:\n";
          	std::cin >> restartEnd;
          	std::cout << "choose stepsize of FGMRES restart range:\n";
          	std::cin >> restartStep;

            // define jacobi iteration range
            int iterStart;
            int iterEnd;
            int iterStep;
            std::cout << "choose start of jacobi iteration range:\n";
          	std::cin >> iterStart;
          	std::cout << "choose end of jacobi iteration range:\n";
          	std::cin >> iterEnd;
          	std::cout << "choose stepsize of jacobi iteration range:\n";
          	std::cin >> iterStep;

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
                solver->FGMRES(A, b, x, tol, i, r, k, &res);

                // write result to logfile
                writeResult(&myfile, &res);

                // free memory
                delete[] x;
                delete[] r;
              }

            }      
          } else {
            std::cout << "file Error" << "\n";
          }

          // end logging
          myfile.close();

          // free memory
          delete[] b;
        }
        break;

      // execute jacobi only  
      case 5:
        {
        	// initialize dimensions
          int nBlock = A->nBlock;
          int n = A->n;
          
          // initialize iterations
          int iterationsLower;
          int iterationsUpper;
          std::cout << "choose lower jacobi iterations:\n";
          std::cin >> iterationsLower;
          std::cout << "choose upper jacobi iterations:\n";
          std::cin >> iterationsUpper;

          // init async block count
          int jacobiAsyncBlockCount;
					std::cout << "choose Async Block Count:\n";
          std::cin >> jacobiAsyncBlockCount;          

          // initialize iterations
          int executions = 1;
          std::cout << "choose execution count:\n";
          std::cin >> executions;

          // compute right side
          double *b = new double[n] {};
          calculateB(nBlock, b);
    
          // allocate and initialize x0, y, r and s
          double *x = new double[n] {}; // initial guess
          double *y = new double[n] {}; // temp for residual and jacobi version without concurrent access
          double *r = new double[n] {}; // rigth side like in first FGMRES iteration
          double *s = new double[n] {}; // store result between lower and upper 

          // allocate experimental vectors
          double *one = new double[n] {};
          double *gUz = new double[n] {};
          double *gLz = new double[n] {};

          for(int i = 0; i < n; i++) {
          	one[i] = 1;
          }

          // compute vector from first FGMRES iteration
          //i dont know if this is a good test
          const char transpose = 'n';
          epsilonfem::EFEM_BLAS::dcsrgemv(&transpose, &n, A->val, A->row, A->col, x, r);
          epsilonfem::EFEM_BLAS::daxpy (n, -1, b, 1, r, 1);
          double g = epsilonfem::EFEM_BLAS::dnrm2 (n, r, 1);
          epsilonfem::EFEM_BLAS::dscal (n, 1.0/g, r, 1);
          

          //messuring execution time
    			std::vector<std::chrono::high_resolution_clock::time_point> jacobi_start;
    			std::vector<std::chrono::high_resolution_clock::time_point> jacobi_between;
    			std::vector<std::chrono::high_resolution_clock::time_point> jacobi_stop;

    
					//setup preconditioning accuracy messurement
    			std::vector<double> normLower;
    			std::vector<double> normUpper;

          //pointer to upper and lower
          void (*lower)(const int, const int, const int, double*, double*, int*, int*, double*) = &lowerLine;
          void (*upper)(const int, const int, const int, double*, double*, int*, int*, double*) = &upperLine;
         


          // repeat jacobi execution
          for(int i = 0; i < executions; i++) {

          	// check what norm is with initial guess
          	//std::cout << "norm Lower befor: " << computeResidualLower(A, one, gLz, y) << "\n";
          	//std::cout << "norm Upper befor: " << computeResidualLower(A, one, gUz, y) << "\n";

						// start Time messurement
          	jacobi_start.push_back(std::chrono::high_resolution_clock::now());

          	// lower jacobi
          	//jacobiLowerSync(A, iterationsLower, one, gLz, y);
          	jacobiLowerHalfSync(A, iterationsLower, one, gLz);
          	//jacobiLowerAsync(A, iterationsLower, r, s);
          	//jacobi(A, iterationsLower, one, gLz, lower, jacobiAsyncBlockCount);
            jacobiLowerDyn(A, iterationsLower, one, gLz, jacobiAsyncBlockCount);

          

          	// checkpoint for time mesuurement
          	jacobi_between.push_back(std::chrono::high_resolution_clock::now());


          	// check what norm is with initial guess
          	//std::cout << "norm befor: " << computeResidualUpper(A, s, x, y) << "\n";

          	// upper jacobi
          	//jacobiUpperSync(A, iterationsUpper, one, gUz, y);
          	//jacobiUpperHalfSync(A, iterationsUpper, one, gUz);
          	//jacobiUpperAsync(A, iterationsUpper, s, x);
          	//jacobi(A, iterationsUpper, one, gUz, upper, jacobiAsyncBlockCount);

            jacobi(A, iterationsUpper, one, gUz, lower, jacobiAsyncBlockCount);



          	// end Time messurement
          	jacobi_stop.push_back(std::chrono::high_resolution_clock::now());


          	// calculate norm
          	normLower.push_back(computeResidualLower(A, one, gLz, y));
          	//normUpper.push_back(computeResidualUpper(A, one, gUz, y));
            normUpper.push_back(computeResidualLower(A, one, gUz, y));


          	delete[] gLz;
          	delete[] gUz;

          	delete[] s;
          	delete[] x;
          	s = new double[n] {};
          	x = new double[n] {};
          	gLz = new double[n] {};
          	gUz = new double[n] {};
          	//std::cout << "\n";
          }


					//calculate times
    			auto duration_jacobi_lower = 0;
    			auto duration_jacobi_upper = 0;

    			//calculate average norm of jacobi
			    double sum_norm_lower = 0;
			    double sum_norm_upper = 0;
    			
    			for (int i = 0; i < jacobi_start.size(); i++) {
    			  auto diff_lower = std::chrono::duration_cast<std::chrono::microseconds>(jacobi_between.at(i) - jacobi_start.at(i)).count();
    			  auto diff_upper = std::chrono::duration_cast<std::chrono::microseconds>(jacobi_stop.at(i) - jacobi_between.at(i)).count();
    			  duration_jacobi_lower += diff_lower;
    			  duration_jacobi_upper += diff_upper;
    			  std::cout << (double)diff_lower / 1000000 << " ";
    			  std::cout << (double)diff_upper / 1000000 << " ";

    			  sum_norm_lower += normLower.at(i);
			      sum_norm_upper += normUpper.at(i);

			      std::cout << normLower.at(i) << " ";
			      std::cout << normUpper.at(i) << "\n";
    			}
  
          //store results in result struct TODO
    			double averageExecTimeLower = ((double)duration_jacobi_lower / executions) / 1000000;
    			double averageExecTimeUpper = ((double)duration_jacobi_upper / executions) / 1000000;
    			double averageNormJacobiLower = sum_norm_lower / executions;
    			double averageNormJacobiUpper = sum_norm_upper / executions;

          //console output
          std::cout << "iterL: " << iterationsLower << " iterU: " << iterationsUpper;
          std::cout << " exec: " << executions << " / residualNorm: ";
          std::cout << averageNormJacobiLower << " / ";
          std::cout << averageNormJacobiUpper << "\n";
          std::cout << "time: ";
          std::cout << averageExecTimeLower << " / ";
          std::cout << averageExecTimeUpper << "\n\n";

          // free memory
          delete[] one;
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
          
          // initialize dimensions
          int nBlock = A->nBlock;
          int n = A->n;

          // init async block count
          int jacobiAsyncBlockCount;
          std::cout << "choose Async Block Count:\n";
          std::cin >> jacobiAsyncBlockCount;  

          // allocate experimental vectors
          double *one = new double[n] {};
          double *gUz = new double[n] {};
          double *gLz = new double[n] {};

          for(int i = 0; i < n; i++) {
            one[i] = 1;
          }

          jacobiLowerDyn(A, 4, one, gLz, jacobiAsyncBlockCount);


          delete[] one;
          delete[] gLz;
          delete[] gUz;
          



          /* // test barrier
          int maxThreadCount = omp_get_max_threads();

          omp_lock_t locks[maxThreadCount];
          int teamMembersFinishedIteration[maxThreadCount] = {};
          int teamBarrier[maxThreadCount] = {};
          int team = 0;
  
          #pragma omp parallel // num_threads(4)
          {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            omp_init_lock(&locks[tid]);  

            omp_set_lock(&locks[team]);
            std::cout << "nthreads: " << nthreads << " tid: " << tid << "\n";
            omp_unset_lock(&locks[team]);

            for(int i = 0; i < 10; i++) {

              // do some

              while(teamBarrier[team] != 0);

              omp_set_lock(&locks[team]);
              if(++teamMembersFinishedIteration[team] < nthreads) {

                std::cout << "enter tid: " << tid << " in: " << teamMembersFinishedIteration[team] << "\n";
                
                omp_unset_lock(&locks[team]);
                while(teamBarrier[team] == 0);
                omp_set_lock(&locks[team]);
              } else {

                std::cout << "enter tid: " << tid << " in: " << teamMembersFinishedIteration[team] << " last\n";

                teamBarrier[team] = 1;
              }

              std::cout << "exit tid: " << tid << " in: " << teamMembersFinishedIteration[team] << "\n";

              if(--teamMembersFinishedIteration[team] == 0) {
                teamBarrier[team] = 0;
                std::cout << "was last\n";
              }
              omp_unset_lock(&locks[team]);
            }
          }
          */

        	/*//std::cout << "env " << std::getenv("OMP_NUM_THREADS") << "\n";
        	std::cout << "maxthreads " << omp_get_max_threads() << "\n";

        	#pragma omp parallel
        	{

        		#pragma omp single
        		std::cout << "maxthreads " << omp_get_num_threads() << "\n";
        	}
        	*/


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