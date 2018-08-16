#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "sparsematrix.h"




// teaming with syncronisation while redistributing
// first test
void jacobi(SparseMatrix *I, const int iterations, double *b, double *x, void (*func)(const int, const int, const int, double*, double*, int*, int*, double*)) {
  
  // initialize parameters
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;

  // initialize OMP parameter
  omp_set_nested(1);
  int maxThreadCount = omp_get_max_threads();
	std::cout << "maxThreadCount: " << maxThreadCount << std::endl;

	// this part is hardcoded for the testmachine it may not work well on other setups
	int sections = 6;
	int teams[6] = {1,1,6,12,24,48};

	for(int i = 0; i < sections; i++) {

		int chunkSize = n / teams[i];

		#pragma omp parallel num_threads(teams[i])
		{
			int tid = omp_get_thread_num();
			int threads = omp_get_num_threads(); 

			int start = tid * chunkSize;
			int end;
			if (tid < teams[i] - 1)
				 end = (tid + 1) * chunkSize;
			else
				 end = n;

			int threadsPerTeam = maxThreadCount / teams[i];

			#pragma omp critical
			{
				std::cout << "tid: " << tid << " threads_active: " << threads << std::endl;
				std::cout << "start: " << start << " stop: " << end << std::endl;
			}

			#pragma omp barrier

			#pragma omp parallel for num_threads(threadsPerTeam)
			for (int k = start; k < end; ++k)
			{
					func(k, nBlock, n, b, x, row, col, val);
			}
		}
	}
}



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
	int diagVal = *(data++);
	
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


// fully syncronized jacobi version
void jacobiLower(SparseMatrix *I, const int iterations, double *b, double *x) {
      
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


void jacobiUpper(SparseMatrix *I, const int iterations, double *b, double *x) {
      
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
	      int diagVal = *(data++);
	
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
      	int diagVal = *(data++);
	
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