#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "sparsematrix.h"


void jacobiLower(SparseMatrix *I, const int iterations, double *b, double *x) {
      
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;
  
  
  #pragma omp parallel
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


void jacobiUpper(SparseMatrix *I, const int iterations, double *b, double *x) {
      
  int n = I->n;
  int nBlock = I->nBlock;
  
  double *val = I->valILU;
  int valSize = I->valSize;
  int *col = I->col;
  int *row = I->row;
  int rowSize = I->rowSize;
  
  
  #pragma omp parallel
  for (int m = 0; m < iterations; m++) {
    
    #pragma omp parallel for schedule(static)
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