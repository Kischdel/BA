#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "blas.h"
#include "sparsematrix.h"
#include "jacobi_par.h"


// "dynamic" blockasynchron jacobi
void jacobiLowerDyn(SparseMatrix *I, const int iterations, double *b, double *x, int asyncBlockCount) {

  // initialize parameters
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;

  int numSections = 1;
  int teamConfig[numSections] = {1}; //, 1, 1, 1};
  teamConfig[0] = asyncBlockCount;


  int maxThreadCount = omp_get_max_threads();

  // arrays to controll execution
  int iterationsPerThread[maxThreadCount] = {};
  int iterationsPerTeam[maxThreadCount] = {};
  int teamMembersFinished[maxThreadCount] = {};
  int teamMembersFinishedIteration[maxThreadCount] = {};
  
  // parameter for barrier
  int teamBarrier[maxThreadCount] = {};
  omp_lock_t locks[maxThreadCount];

  // number of teams finished
  int teamsFinished = 0;

  // initialize abort condition
  bool proceedToNextSection = false;



  #pragma omp parallel num_threads(maxThreadCount)
  {
    // initialize omp parameter
    int tid = omp_get_thread_num();
    int numThreads = omp_get_num_threads();
    omp_init_lock(&locks[tid]);


    // loop for sections
    for (int i = 0; i < numSections; i++) {

      // initialize parameter that define worksplitting
      int numTeams = teamConfig[i];

      int threadsPerTeam = numThreads / numTeams;
      int team = tid / threadsPerTeam;

      int leader = team * threadsPerTeam;

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

      /*// temp console output
      #pragma omp critical
      {
        std::cout << "tid: " << tid << " team: " << team << " leader: " << leader << " section: " << i;
        std::cout << " start: " << start << " stop: " << end << std::endl;
      }
      */


      // blockasynchron loop
      do {

        // progress lines
        for (int j = start; j < end; j++) {
          lowerLine(j, nBlock, n, b, x, row, col, val);
        }

        iterationsPerThread[tid]++;

        /*#pragma omp critical
        {
            std::cout << "tid: " << tid << " team: " << team << " leader: " << leader << " iterations executed " << iterationsPerThread[tid] << std::endl;
        }
        */

        // update thread iter count and check if min iterations reached
        if (tid == leader && iterationsPerThread[tid] == iterations) {

          #pragma omp atomic
          teamsFinished++;

          /*#pragma omp critical
          {
            std::cout << "tid: " << tid << " team: " << team << " leader: " << leader << " teamsFinished: " << teamsFinished << std::endl;
          }
          */

          if (teamsFinished >= numTeams / 2) {

            proceedToNextSection = true;
          }
        }

        // custom Barrier
        while (teamBarrier[team] != 0);

        omp_set_lock(&locks[team]);
        if (++teamMembersFinishedIteration[team] < threadsPerTeam) {

          omp_unset_lock(&locks[team]);
          while (teamBarrier[team] == 0);
          omp_set_lock(&locks[team]);

        } else {

          teamBarrier[team] = 1;
        }

        if (--teamMembersFinishedIteration[team] == 0) {
          
          teamBarrier[team] = 0; 
        }
        omp_unset_lock(&locks[team]);
        // end custom Barrier

      } while (!proceedToNextSection);


      // barrier before preparation for next section
      #pragma omp barrier

      // reset parameters
      teamMembersFinished[tid] = 0;


      
      // end of section barrier
      #pragma omp barrier

    } // end Sections

    #pragma omp barrier
    // cleanup
    omp_destroy_lock(&locks[tid]);
  }
}



// teaming with syncronisation while redistributing teams
void jacobi(SparseMatrix *I, const int iterations, double *b, double *x, void (*func)(const int, const int, const int, double*, double*, int*, int*, double*), int asyncBlockCount) {
  
  // initialize parameters
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;

  // parameter for y tmp vector
  // double *y = new double[n] {};

  // initialize OMP parameter
  omp_set_nested(1);
  int maxThreadCount = omp_get_max_threads();
	//std::cout << "maxThreadCount: " << maxThreadCount << std::endl;

	// this part is hardcoded for the testmachine it may not work well on other setups
  int sections = 1;
  int iterSections = iterations;
	int teams[sections] = {6}; //, 3, 8, 24};
  // temp for manuell block asyncronus execution without Blockrestructuration
  teams[0] = asyncBlockCount;

  // for every section
  for(int i = 0; i < sections; i++)
  {
    
    // split into teams
    #pragma omp parallel num_threads(teams[i])
    {

      int parent_tid = omp_get_thread_num();
      int threads = omp_get_num_threads(); 

      int chunkRest = n % teams[i];
      int tidOffset = threads - chunkRest;
      int chunkSize = n / teams[i];
  
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

	 		int threadsPerTeam = maxThreadCount / teams[i];
  
      /*
	 		#pragma omp critical
	 		{
	 			std::cout << "tid: " << parent_tid << " threads_active: " << threads;
	 			std::cout << " start: " << start << " stop: " << end << std::endl;
	 		}

  
	 		#pragma omp barrier
      */

      //iterations in jacobi per section
      for (int j = 0; j < iterSections; j++)
      {
        //create child threads in each team to solve lines
        if (func == &lowerLine) {
          #pragma omp parallel for num_threads(threadsPerTeam)
          for (int k = start; k < end; ++k)
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
        
      }
      
      //#pragma omp barrier

      /*
      #pragma omp single
      {
        if(func == &lowerLine)
          std::cout << "section: " << i << " normLower: " << computeResidualLower(I, b, x, y) << "\n";
        else
          std::cout << "section: " << i << " normUpper: " << computeResidualUpper(I, b, x, y) << "\n";
      }
      */

    } //end #pragma parallel 
  } //end for section
  
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
     	x_at_lineIndex -= *(data++) * x[*(dataCol++)];
	       
    	//x_at_i /= *data;	// not needed, because the lower diagonal is 1
	        
  } else {
	      
   	if (notFirstRow)
     	x_at_lineIndex -= *(data++) * x[*(dataCol++)];
   	x_at_lineIndex -= *(data++) * x[*(dataCol++)];
	        
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
	    x_at_lineIndex -= *(data++) * x[*(dataCol++)];
        	
	  x_at_lineIndex /= diagVal;
        	
	} else {
      	
	  if (notLastRow)
	    x_at_lineIndex -= *(data++) * x[*(dataCol++)];
	  x_at_lineIndex -= *(data++) * x[*(dataCol++)];
        	
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
            x_at_i -= *(data++) * x0[*(dataCol++)];
          
          //x_at_i /= *data;
          
        } else {
        
          if (notFirstRow)
            x_at_i -= *(data++) * x0[*(dataCol++)];
          x_at_i -= *(data++) * x0[*(dataCol++)];
          
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
            x_at_i -= *(data++) * x0[*(dataCol++)];
          
          x_at_i /= diagVal;
          
        } else {
        
          if (notLastRow)
            x_at_i -= *(data++) * x0[*(dataCol++)];
          x_at_i -= *(data++) * x0[*(dataCol++)];
          
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
          	x_at_i -= *(data++) * x[*(dataCol++)];
	        
        	//x_at_i /= *data;
	        
      	} else {
	      
        	if (notFirstRow)
          	x_at_i -= *(data++) * x[*(dataCol++)];
        	x_at_i -= *(data++) * x[*(dataCol++)];
	        
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
          	x_at_i -= *(data++) * x[*(dataCol++)];
	        
        	x_at_i /= diagVal;
	        
      	} else {
	      
        	if (notLastRow)
          	x_at_i -= *(data++) * x[*(dataCol++)];
        	x_at_i -= *(data++) * x[*(dataCol++)];
	        
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
            x_at_i -= *(data++) * x[*(dataCol++)];
          
          //x_at_i /= *data;
          
        } else {
        
          if (notFirstRow)
            x_at_i -= *(data++) * x[*(dataCol++)];
          x_at_i -= *(data++) * x[*(dataCol++)];
          
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
            x_at_i -= *(data++) * x[*(dataCol++)];
          
          x_at_i /= diagVal;
          
        } else {
        
          if (notLastRow)
            x_at_i -= *(data++) * x[*(dataCol++)];
          x_at_i -= *(data++) * x[*(dataCol++)];
          
          x_at_i /= diagVal; 
        }
        x[i] = x_at_i;
      }  
    }
  }
}
