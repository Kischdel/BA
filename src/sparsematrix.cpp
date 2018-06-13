#include "sparsematrix.h"
#include <vector>
#include <algorithm>
#include <iostream>

void SparseMatrix::calculateEntrys(int nBlock, std::vector<ent*> *list) {
  
  // diag
  for(int k = 0; k < nBlock * nBlock; k++) {
    
    ent *newE = (ent*) malloc(sizeof(struct entry));
    newE->i = k;
    newE->j = k;
    newE->val = 4;
    (*list).push_back(newE); 
  }
  
  // sidediag
  for(int k = 0; k < nBlock * nBlock; k++) {
    // over diag
    if(k % nBlock != nBlock - 1) {
    
      ent *newE = (ent*) malloc(sizeof(struct entry));
      newE->i = k;
      newE->j = k + 1;
      newE->val = -1;
      (*list).push_back(newE);
    }
    
    // under diag
    if(k % nBlock != 0) {
    
      ent *newE = (ent*) malloc(sizeof(struct entry));
      newE->i = k;
      newE->j = k - 1;
      newE->val = -1;
      (*list).push_back(newE);
    } 
  }
  
  // side Blocks
  for(int k = 0; k < nBlock * (nBlock - 1); k++) {
  
    ent *newE = (ent*) malloc(sizeof(struct entry));
    newE->i = k;
    newE->j = k + nBlock;
    newE->val = -1;
    (*list).push_back(newE);
  }
  
  for(int k = 0; k < nBlock * (nBlock - 1); k++) {
  
    ent *newE = (ent*) malloc(sizeof(struct entry));
    newE->i = k + nBlock;
    newE->j = k;
    newE->val = -1;
    list->push_back(newE);
  }
  
  
  std::sort(list->begin(), list->end(), SparseMatrix::PComp);
}

SparseMatrix::SparseMatrix(int nBlock) {
  this->n = nBlock*nBlock;
  this->nBlock = nBlock;
  
  std::vector<ent*> list;
  calculateEntrys(nBlock, &list);
  this->valSize = list.size();
  this->rowSize = n + 1;
  val = new double[valSize];
  col = new int[valSize];
  row = new int[rowSize];
  
  row[0] = 0;
  row[n] = valSize;
  int lastRowEntry = 0;
  int rowCount = 0;
  
  for(int k = 0; k < valSize; k++) {
    
    val[k] = list.at(k)->val;
    col[k] = list.at(k)->j;
    
    if(list.at(k)->i != lastRowEntry) {
      rowCount++;
      row[rowCount] = k;
      lastRowEntry = list.at(k)->i; 
    }
    free(list.at(k));
  }
}

SparseMatrix::~SparseMatrix() {
  delete[] val;
  delete[] col;
  delete[] row;
}