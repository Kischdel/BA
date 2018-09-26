#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "blas.h"
#include "sparsematrix.h"
#include "jacobi_par.h"


// "dynamic" blockasynchron jacobi with busy wait
void jacobiDynBusy(SparseMatrix *I, double *b, double *x, void (*func)(const int, const int, const int, double*, double*, int*, int*, double*), int blockCount, int iterations) {

  // initialize parameters
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;

  int numThreads = omp_get_max_threads();
  int threadsPerTeam = numThreads / blockCount;

  // arrays to controll execution
  int iterationsPerThread[numThreads] = {};
  int teamMembersFinishedIteration[blockCount] = {};
  
  // parameter for barrier
  int teamBarrier[blockCount] = {};
  omp_lock_t locks[blockCount];
  for (int i = 0; i < blockCount; i++)
    omp_init_lock(&locks[i]);

  // number of teams finished
  int teamsFinished = 0;

  // initialize abort condition
  int finished[blockCount] = {};


  #pragma omp parallel
  {
    // initialize omp parameter
    int tid = omp_get_thread_num();
    int numThreads = omp_get_num_threads();
    

    int chunkRest = n % numThreads;
    int tidOffset = numThreads - chunkRest;
    int chunkSize = n / numThreads;
  
    int start;
    int end;
    int offset = tidOffset * chunkSize;


    if (tid < tidOffset) {

      start = tid * chunkSize;
      end = start + chunkSize;

    } else {

      chunkSize++;
      start = offset + ((tid - tidOffset) * chunkSize);
      end = start + chunkSize;
    }

    int team = tid / threadsPerTeam;

    /* debug output
    #pragma omp critical
    {
      std::cout << "tid: " << tid << " team: " << team  << "\n";
    }
    #pragma omp barrier
    //*/


    // blockasynchron loop
    do {

      // progress lines
      if (func == &lowerLine) {
        for (int j = start; j < end; ++j)
        {
          func(j, nBlock, n, b, x, row, col, val);
        }
      } else {

        for (int j = end - 1; j >= start; --j)
        {
          func(j, nBlock, n, b, x, row, col, val);
        }
      }

      iterationsPerThread[tid]++;
      

      // custom Barrier end of iteration      
      while (teamBarrier[team] != 0);

      omp_set_lock(&locks[team]);
      // debug output
      //std::cout << "tid: " << tid << " team: " << team << " iterF: " << iterationsPerThread[tid] << " B: " << finished[team] << "\n";
      if (++teamMembersFinishedIteration[team] < threadsPerTeam) {
        //debug output
        //std::cout << teamMembersFinishedIteration[team] << "\n";
        omp_unset_lock(&locks[team]);
        while (teamBarrier[team] == 0);
        omp_set_lock(&locks[team]);

      } else {
        // debug output
        //std::cout << "tid: " << tid << " team: " << team << " count to last thread correct\n";
        
        teamBarrier[team] = 1;

        // check if min iterations reached
        // it is the job of the last thread to enter, because if done earlier it might cause a deadlock
        if (iterationsPerThread[tid] == iterations) {

          #pragma omp atomic
          teamsFinished++;
        }
        // if enough teams have finished abort
        if (teamsFinished >= blockCount / 3 && teamsFinished >= 1 ) {

            finished[team] = 1;
        }
      }

      if (--teamMembersFinishedIteration[team] == 0) {
          
        teamBarrier[team] = 0; 
      }

      // debug output
      //std::cout << "tid: " << tid << " team: " << team << " iterF: " << iterationsPerThread[tid] << " teamF: " << teamsFinished << " A: " << finished[team] << "\n";

      omp_unset_lock(&locks[team]);
      // end custom Barrier

    } while (finished[team] == 0);

    // cleanup
    omp_destroy_lock(&locks[team]);
  } // end parallel section
}



// teaming with syncronisation while redistributing teams
void jacobi(SparseMatrix *I, double *b, double *x, void (*func)(const int, const int, const int, double*, double*, int*, int*, double*), int secNum, int *secConf, int *secIter) {
  
  // initialize parameters
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;

  // parameter for y tmp vector //only needed for inline norm computation
  //double *y = new double[n] {};

  // initialize OMP parameter
  omp_set_nested(1);
  int maxThreadCount = omp_get_max_threads();
	//std::cout << "maxThreadCount: " << maxThreadCount << std::endl;

  // initialize abort condition
  int teamsFinished = 0;
  bool proceedToNextSection = false;
  int iterationsPerTeam[maxThreadCount] = {};

  // for every section
  for(int i = 0; i < secNum; i++)
  {
    
    // split into teams
    #pragma omp parallel num_threads(secConf[i])
    {

      int parent_tid = omp_get_thread_num();
      int threads = omp_get_num_threads(); 

      int chunkRest = n % secConf[i];
      int tidOffset = threads - chunkRest;
      int chunkSize = n / secConf[i];
  
      int start;
      int end;
      int offset = tidOffset * chunkSize;

      if (parent_tid < tidOffset) {

        start = parent_tid * chunkSize;
        end = start + chunkSize;

      } else {

        chunkSize++;
        start = offset + ((parent_tid - tidOffset) * chunkSize);
        end = start + chunkSize;
      }

	 		int threadsPerTeam = maxThreadCount / secConf[i];
  
      /*
	 		#pragma omp critical
	 		{
	 			std::cout << "tid: " << parent_tid << " threads_active: " << threads;
	 			std::cout << " start: " << start << " stop: " << end << std::endl;
	 		}
      
  
	 		#pragma omp barrier
      */

      //iterations in jacobi per section
      do {
        //create child threads in each team to solve lines
        if (func == &lowerLine) {
          #pragma omp parallel for num_threads(threadsPerTeam)
          for (int k = start; k < end; ++k)
          {
            /*
            #pragma omp critical
            {
              std::cout << "tid: " << parent_tid << " / " << omp_get_thread_num() << " threads_active: " << omp_get_num_threads();
              std::cout << " iter: " << iterationsPerThread[parent_tid];
              std::cout << " line: " << k << std::endl;
            }
            */
            func(k, nBlock, n, b, x, row, col, val);
          }
        } else {
          #pragma omp parallel for num_threads(threadsPerTeam)
          for (int k = end - 1; k >= start; --k)
          //for (int k = start; k < end; ++k)
          {
            /*
            #pragma omp critical
            {
              std::cout << "tid: " << parent_tid << " / " << omp_get_thread_num() << " threads_active: " << omp_get_num_threads();
              std::cout << " iter: " << j;
              std::cout << " line: " << k << std::endl;
            }
            */
            func(k, nBlock, n, b, x, row, col, val);
          }
        }
        
        iterationsPerTeam[parent_tid]++;

        /*
        #pragma omp critical
        {
            std::cout << "tid: " << parent_tid << " iterations executed " << iterationsPerTeam[parent_tid] << std::endl;
        }
        */

        // update thread iter count and check if min iterations reached
        if (iterationsPerTeam[parent_tid] == secIter[i]) {

          #pragma omp atomic
          teamsFinished++;

          /*
          #pragma omp critical
          {
            std::cout << "tid: " << parent_tid << " teamsFinished: " << teamsFinished << std::endl;
          }
          */

          if (teamsFinished >= secConf[i] / 3) {

            proceedToNextSection = true;
            /*
            #pragma omp critical
            {
              std::cout << "tid: " << parent_tid << " abort section" << std::endl;
            }
            */
          }
        }

      } while (!proceedToNextSection);
      
      //#pragma omp barrier

    iterationsPerTeam[parent_tid] = 0;

    } //end #pragma parallel 

    // reset parameter
    proceedToNextSection = false;
    teamsFinished = 0;

    /* inline residuum
    if(func == &lowerLine)
      std::cout << "section: " << i << " normLower: " << computeResidualLower(I, b, x, y) << "\n";
    else
      std::cout << "section: " << i << " normUpper: " << computeResidualUpper(I, b, x, y) << "\n";
    */

  } //end for section
  
  omp_set_nested(0);
  //free memory
  //delete[] y;
}


// function to compute a single jacobi line
void lowerLine(const int lineIndex, const int nBlock, const int n, double *b, double *x, int *row, int *col, double *val) {

	bool firstBlockRow = (lineIndex < nBlock);
  bool notFirstRow = (lineIndex % nBlock != 0);
  int rowIndex = row[lineIndex];
  double x_at_lineIndex = b[lineIndex];
  double *data = val + rowIndex;
  int *dataCol = col + rowIndex;
	      
	      
  if (firstBlockRow) {
	   
   	if (lineIndex != 0)
     	x_at_lineIndex -= *data * x[*dataCol];
	       
    	//x_at_i /= *data;	// not needed, because the lower diagonal is 1
	        
  } else {
	      
   	if (notFirstRow)
     	x_at_lineIndex -= *(data++) * x[*(dataCol++)];
   	x_at_lineIndex -= *data * x[*dataCol];
	        
   	//x_at_i /= *data;  // not needed, because the lower diagonal is 1
  }
  x[lineIndex] = x_at_lineIndex;
}


void upperLine(const int lineIndex, const int nBlock, const int n, double *b, double *x, int *row, int *col, double *val) {

	bool firstBlockRow = (lineIndex < nBlock);
  bool notFirstRow = (lineIndex % nBlock != 0);
  bool lastBlockRow = (lineIndex >= n - nBlock);
	bool notLastRow = (lineIndex % nBlock != nBlock - 1);
  int rowIndex = row[lineIndex];
  double x_at_lineIndex = b[lineIndex];
  double *data;
	int *dataCol;
      	
	
	// calculate where the diagonal element is located
	int offsetUpper = 0;
	
	if (firstBlockRow) {
	
	  if (lineIndex != 0)
	    offsetUpper = 1;
	
	} else {
	
	  offsetUpper = 1;
	  if (notFirstRow)
	    offsetUpper = 2;
	}
	
	data = val + rowIndex + offsetUpper;
	dataCol = col + rowIndex + offsetUpper + 1;
	
	// store diagonal Value for division
	double diagVal = *(data++);
	
	if (lastBlockRow) {
        	
	  if (notLastRow)
	    x_at_lineIndex -= *data * x[*dataCol];
        	
	  x_at_lineIndex /= diagVal;
        	
	} else {
      	
	  if (notLastRow)
	    x_at_lineIndex -= *(data++) * x[*(dataCol++)];
	  x_at_lineIndex -= *data * x[*dataCol];
        	
	  x_at_lineIndex /= diagVal; 
	}
	x[lineIndex] = x_at_lineIndex;
}


// fully syncronized jacobi version without concurrent access to x
void jacobiLowerSync(SparseMatrix *I, const int iterations, double *b, double *x0, double *x) {
      
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;

  #pragma omp parallel
  {
    for (int m = 0; m < iterations; m++) {
      
      #pragma omp for schedule(static)
      for (int i = 0; i < n; i++) {
        
        bool firstBlockRow = (i < nBlock);
        bool notFirstRow = (i % nBlock != 0);
        int rowIndex = row[i];
        double x_at_i = b[i];
        double *data = val + rowIndex;
        int *dataCol = col + rowIndex;
        
        
        if (firstBlockRow) {
          
          if (i != 0)
            x_at_i -= *data * x0[*dataCol];
          
          //x_at_i /= *data;
          
        } else {
        
          if (notFirstRow)
            x_at_i -= *(data++) * x0[*(dataCol++)];
          x_at_i -= *data * x0[*dataCol];
          
          //x_at_i /= *data; 
        }
        x[i] = x_at_i;
      }
      #pragma omp for schedule(static)
      for (int i = 0; i < n; i++) {

        x0[i] = x[i];
      }
    }
  }
}


void jacobiUpperSync(SparseMatrix *I, const int iterations, double *b, double *x0, double *x) {
      
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;
  
  
  #pragma omp parallel
  {
    for (int m = 0; m < iterations; m++) {
      
      #pragma omp for schedule(static)
      for (int i = 0; i < n; i++) {
        
        bool firstBlockRow = (i < nBlock);
        bool notFirstRow = (i % nBlock != 0);
        bool lastBlockRow = (i >= n - nBlock);
        bool notLastRow = (i % nBlock != nBlock - 1);
        int rowIndex = row[i];
        double x_at_i = b[i];
        double *data;
        int *dataCol;
        
  
        // calculate where the diagonal element is located
        int offsetUpper = 0;
  
        if (firstBlockRow) {
  
          if (i != 0)
            offsetUpper = 1;
  
        } else {
  
          offsetUpper = 1;
          if (notFirstRow)
            offsetUpper = 2;
        }
  
        data = val + rowIndex + offsetUpper;
        dataCol = col + rowIndex + offsetUpper + 1;

        // store diagonal Value for division
        double diagVal = *(data++);
  
        if (lastBlockRow) {
          
          if (notLastRow)
            x_at_i -= *data * x0[*dataCol];
          
          x_at_i /= diagVal;
          
        } else {
        
          if (notLastRow)
            x_at_i -= *(data++) * x0[*(dataCol++)];
          x_at_i -= *data * x0[*dataCol];
          
          x_at_i /= diagVal; 
        }
        x[i] = x_at_i;
      } 
      #pragma omp for schedule(static)
      for (int i = 0; i < n; i++) {

        x0[i] = x[i];
      }
    }
  }
}

// fully syncronized jacobi version without concurrent access to x
void parallelOverheadLower(SparseMatrix *I, const int iterations, double *b, double *x) {
      
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;

  #pragma omp parallel
  {
    for (int m = 0; m < iterations; m++) {
      
      #pragma omp for schedule(static)
      for (int i = 0; i < n; i++) {
      }
    }
  }
}

void parallelOverheadUpper(SparseMatrix *I, const int iterations, double *b, double *x) {
      
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;
  
  
  #pragma omp parallel
  {
    for (int m = 0; m < iterations; m++) {
      
      #pragma omp for schedule(static)
      for (int i = n -1; i >= 0; i--) {
      }
    }
  }
}


// asyncron jacobi version
void jacobiLowerAsync(SparseMatrix *I, const int iterations, double *b, double *x) {
      
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;
  
  
  #pragma omp parallel
  {
  	for (int m = 0; m < iterations; m++) {
	    
    	#pragma omp for schedule(static) nowait
    	for (int i = 0; i < n; i++) {
	      
      	bool firstBlockRow = (i < nBlock);
      	bool notFirstRow = (i % nBlock != 0);
      	int rowIndex = row[i];
      	double x_at_i = b[i];
      	double *data = val + rowIndex;
      	int *dataCol = col + rowIndex;
	      
	      
      	if (firstBlockRow) {
	        
        	if (i != 0)
          	x_at_i -= *data * x[*dataCol];
	        
        	//x_at_i /= *data;
	        
      	} else {
	      
        	if (notFirstRow)
          	x_at_i -= *(data++) * x[*(dataCol++)];
        	x_at_i -= *data * x[*dataCol];
	        
        	//x_at_i /= *data; 
      	}
      	x[i] = x_at_i;
    	}  
	  }
	}
}


void jacobiUpperAsync(SparseMatrix *I, const int iterations, double *b, double *x) {
      
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;
  
  
  #pragma omp parallel
  {
  	for (int m = 0; m < iterations; m++) {
	    
    	#pragma omp for schedule(static) nowait
    	for (int i = n - 1; i >= 0; i--) {
	      
      	bool firstBlockRow = (i < nBlock);
      	bool notFirstRow = (i % nBlock != 0);
      	bool lastBlockRow = (i >= n - nBlock);
      	bool notLastRow = (i % nBlock != nBlock - 1);
      	int rowIndex = row[i];
      	double x_at_i = b[i];
      	double *data;
      	int *dataCol;
	      
	
      	// calculate where the diagonal element is located
      	int offsetUpper = 0;
	
      	if (firstBlockRow) {
	
        	if (i != 0)
          	offsetUpper = 1;
	
      	} else {
	
        	offsetUpper = 1;
        	if (notFirstRow)
          	offsetUpper = 2;
      	}
	
      	data = val + rowIndex + offsetUpper;
      	dataCol = col + rowIndex + offsetUpper + 1;
	
      	// store diagonal Value for division
      	double diagVal = *(data++);
	
      	if (lastBlockRow) {
	        
        	if (notLastRow)
          	x_at_i -= *data * x[*dataCol];
	        
        	x_at_i /= diagVal;
	        
      	} else {
	      
        	if (notLastRow)
          	x_at_i -= *(data++) * x[*(dataCol++)];
        	x_at_i -= *data * x[*dataCol];
	        
        	x_at_i /= diagVal; 
      	}
      	x[i] = x_at_i;
    	}  
	  }
	}
}


double computeResidualLower(SparseMatrix* I, double *b, double *x, double *residual) {
      
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;
      
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {

    bool firstBlockRow = (i < nBlock);
    bool notFirstRow = (i % nBlock != 0);
    int rowIndex = row[i];
    //double x_at_i = b[i];
    double *data = val + rowIndex;
    int *dataCol = col + rowIndex;
    residual[i] = 0;

    
    if (firstBlockRow) {
      
      if (i != 0)
        residual[i] += *(data++) * x[*(dataCol++)];
      
      residual[i] += x[*dataCol];
      
    } else {
    
      if (notFirstRow)
        residual[i] += *(data++) * x[*(dataCol++)];

      residual[i] += *(data++) * x[*(dataCol++)];
      residual[i] += x[*dataCol]; 
    }
  }

  // compute residual and norm
  epsilonfem::EFEM_BLAS::daxpy (n, -1, b, 1, residual, 1);
  double norm = epsilonfem::EFEM_BLAS::dnrm2 (n, residual, 1); 

  return norm;
}


double computeResidualUpper(SparseMatrix* I, double *b, double *x, double *residual) {
      
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;
  
  
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
  
    bool firstBlockRow = (i < nBlock);
    bool notFirstRow = (i % nBlock != 0);
    bool lastBlockRow = (i >= n - nBlock);
    bool notLastRow = (i % nBlock != nBlock - 1);
    int rowIndex = row[i];
    //double x_at_i = b[i];
    double *data;
    int *dataCol;
    residual[i] = 0;
      

    // calculate where the diagonal element is located
    int offsetUpper = 0;

    if (firstBlockRow) {

      if (i != 0)
        offsetUpper = 1;

    } else {

      offsetUpper = 1;
      if (notFirstRow)
        offsetUpper = 2;
    }

    data = val + rowIndex + offsetUpper;
    dataCol = col + rowIndex + offsetUpper;

    // compute for diag value
    residual[i] += *(data++) * x[*(dataCol++)];

    if (lastBlockRow) {
      
      if (notLastRow)
        residual[i] += *data * x[*dataCol];
      
    } else {
    
      if (notLastRow)
        residual[i] += *(data++) * x[*(dataCol++)];
      residual[i] += *data * x[*dataCol];
    }
  }  

  // compute residual and norm
  epsilonfem::EFEM_BLAS::daxpy (n, -1, b, 1, residual, 1);
  double norm = epsilonfem::EFEM_BLAS::dnrm2 (n, residual, 1); 

  return norm;
}


void simpleJacobi(SparseMatrix* I, const int iterations, double *b, double *x0, double *x) {

  double res[I->n];
  int count = 0;
  double tol = 1.0e-06;
  double act = 1.0;

  for(int m = 0; m < iterations; m++) {
  //while(act > tol) {

    for(int i = 0; i < I->n; i++) {
      x[i] = b[i];

      for(int j = 0; j < I->n; j++) {
        
        if(i != j) {
          x[i] -= I->getValAt(i,j) * x0[j];
        }
      }
      x[i] /= I->getValAt(i,i); 
    }

    for(int i = 0; i < I->n; i++) {
      x0[i] = x[i];
    }

    act = simpleResidual(I, b, x, res);
    std::cout << "iter: " << ++count << " res: " << act << "\n";
  }
}


void simpleJacobiLower(SparseMatrix* I, const int iterations, double *b, double *x0, double *x) {
  
  double res[I->n];
  int count = 0;
  double tol = 1.0e-06;
  double act = 1.0;

  for(int m = 0; m < iterations; m++) {

    for(int i = 0; i < I->n; i++) {
      x[i] = b[i];

      for(int j = 0; j < I->n; j++) {
        
        if(i != j) {
          x[i] -= I->getLowerILUValAt(i,j) * x0[j];
        }
      }
      x[i] /= I->getLowerILUValAt(i,i); 
    }

    for(int i = 0; i < I->n; i++) {
      x0[i] = x[i];
    }

    act = simpleResidualLower(I, b, x, res);
    std::cout << "iter: " << ++count << " res: " << act << "\n";
  }
}


void simpleJacobiUpper(SparseMatrix* I, const int iterations, double *b, double *x0, double *x) {

  double res[I->n];
  int count = 0;
  double tol = 1.0e-06;
  double act = 1.0;

  for(int m = 0; m < iterations; m++) {

    for(int i = I->n - 1; i >= 0; i--) {
      x[i] = b[i];

      for(int j = 0; j < I->n; j++) {
        
        if(i != j) {
          x[i] -= I->getUpperILUValAt(i,j) * x0[j];
        }
      }
      x[i] /= I->getUpperILUValAt(i,i); 
    }

    act = simpleResidualUpper(I, b, x, res);
    std::cout << "iter: " << ++count << " res: " << act << "\n";
  }
}


double simpleResidual(SparseMatrix *I, double *b, double *x, double *residual) {

  for (int i = 0; i < I->n; i++) {

    residual[i] = 0;
    
    for (int j = 0; j < I->n; j++) {
      residual[i] += I->getValAt(i,j) * x[j];
    }
  }

  // compute residual and norm
  epsilonfem::EFEM_BLAS::daxpy (I->n, -1, b, 1, residual, 1);
  double norm = epsilonfem::EFEM_BLAS::dnrm2 (I->n, residual, 1); 

  return norm;

}


double simpleResidualLower(SparseMatrix *I, double *b, double *x, double *residual) {
  
  for (int i = 0; i < I->n; i++) {

    residual[i] = 0;
    
    for (int j = 0; j < I->n; j++) {
      residual[i] += I->getLowerILUValAt(i,j) * x[j];
    }
  }

  // compute residual and norm
  epsilonfem::EFEM_BLAS::daxpy (I->n, -1, b, 1, residual, 1);
  double norm = epsilonfem::EFEM_BLAS::dnrm2 (I->n, residual, 1); 

  return norm;
}


double simpleResidualUpper(SparseMatrix *I, double *b, double *x, double *residual) {
  
  for (int i = 0; i < I->n; i++) {

    residual[i] = 0;
    
    for (int j = 0; j < I->n; j++) {
      residual[i] += I->getUpperILUValAt(i,j) * x[j];
    }
  }

  // compute residual and norm
  epsilonfem::EFEM_BLAS::daxpy (I->n, -1, b, 1, residual, 1);
  double norm = epsilonfem::EFEM_BLAS::dnrm2 (I->n, residual, 1); 

  return norm;
}


// fully syncronized jacobi version with concurrent access to x
void jacobiLowerHalfSync(SparseMatrix *I, const int iterations, double *b, double *x) {
      
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;
  
  
  #pragma omp parallel
  {
    for (int m = 0; m < iterations; m++) {
      
      #pragma omp for schedule(static)
      for (int i = 0; i < n; i++) {
        
        bool firstBlockRow = (i < nBlock);
        bool notFirstRow = (i % nBlock != 0);
        int rowIndex = row[i];
        double x_at_i = b[i];
        double *data = val + rowIndex;
        int *dataCol = col + rowIndex;
        
        
        if (firstBlockRow) {
          
          if (i != 0)
            x_at_i -= *data * x[*dataCol];
          
          //x_at_i /= *data;
          
        } else {
        
          if (notFirstRow)
            x_at_i -= *(data++) * x[*(dataCol++)];
          x_at_i -= *data * x[*dataCol];
          
          //x_at_i /= *data; 
        }
        x[i] = x_at_i;
      }  
    }
  }
}


void jacobiUpperHalfSync(SparseMatrix *I, const int iterations, double *b, double *x) {
      
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;
  
  
  #pragma omp parallel
  {
    for (int m = 0; m < iterations; m++) {
      
      #pragma omp for schedule(static)
      for (int i = n -1; i >= 0; i--) {
        
        bool firstBlockRow = (i < nBlock);
        bool notFirstRow = (i % nBlock != 0);
        bool lastBlockRow = (i >= n - nBlock);
        bool notLastRow = (i % nBlock != nBlock - 1);
        int rowIndex = row[i];
        double x_at_i = b[i];
        double *data;
        int *dataCol;
        
  
        // calculate where the diagonal element is located
        int offsetUpper = 0;
  
        if (firstBlockRow) {
  
          if (i != 0)
            offsetUpper = 1;
  
        } else {
  
          offsetUpper = 1;
          if (notFirstRow)
            offsetUpper = 2;
        }
  
        data = val + rowIndex + offsetUpper;
        dataCol = col + rowIndex + offsetUpper + 1;
  
        // store diagonal Value for division
        double diagVal = *(data++);
  
        if (lastBlockRow) {
          
          if (notLastRow)
            x_at_i -= *data * x[*dataCol];
          
          x_at_i /= diagVal;
          
        } else {
        
          if (notLastRow)
            x_at_i -= *(data++) * x[*(dataCol++)];
          x_at_i -= *data * x[*dataCol];
          
          x_at_i /= diagVal; 
        }
        x[i] = x_at_i;
      }  
    }
  }
}
