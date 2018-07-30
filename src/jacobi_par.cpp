#include <stdio.h>
#include <iostream>
#include "sparsematrix.h"


void jacobiLower(SparseMatrix *I, const int iterations, double *b, double *x) {
      
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->val;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;
  
  
  for (int m = 0; m < iterations; m++) {
    
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
        
        x_at_i /= *data;
        
      } else {
      
        if (notFirstRow)
          x_at_i -= *(data++) * x[*(dataCol++)];
        x_at_i -= *(data++) * x[*(dataCol++)];
        
        x_at_i /= *data; 
      }
      x[i] = x_at_i;
      
      
      
      /*
      std::cout << "index row: " << rowIndex;
      std::cout << " data array: " << data[0];
      std::cout << " col array: " << dataCol[0];
      
      std::cout << "\n";
      */
    }  
  }
}