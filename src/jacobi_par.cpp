#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "blas.h"
#include "sparsematrix.h"
#include "jacobi_par.h"



// teaming with syncronisation while redistributing teams
// this version has overhead for function calls
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
      #pragma omp single
      {
        epsilonfem::EFEM_BLAS::dcopy (n, x, 1, x0, 1);
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
      #pragma omp single
      {
        epsilonfem::EFEM_BLAS::dcopy (n, x, 1, x0, 1); 
      }
    }
  }
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

    for(int i = 0; i < I->n; i++) {
      x[i] = b[i];

      for(int j = 0; j < I->n; j++) {
        
        if(i != j) {
          x[i] -= I->getUpperILUValAt(i,j) * x0[j];
        }
      }
      std::cout << "befor division: " << x[i] << "\n";
      x[i] /= I->getUpperILUValAt(i,i); 
    }

    for(int i = 0; i < I->n; i++) {
      x0[i] = x[i];
      std::cout << "new_x: " << x[i] << "\n";
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