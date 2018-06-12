#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>

void run_test() {
  
  omp_set_nested(1);
  
  int parent;
  int quit = 0;

  #pragma omp parallel for num_threads(2) private(parent)
  for (int i = 0; i < 4; i++) {
    
    parent = omp_get_thread_num();
    printf("outer loop: thread: %d\n", parent);
    
    #pragma omp parallel for // num_threads(8) dont need that shit
    for (int j = 0; j < 8; j++) {
      
      printf("parent: %d, act_thread: %d\n", parent, omp_get_thread_num());
    
    }
  
  }
}



void run_test2() {

  omp_set_nested(1);
  int next = 0;
  int quit = 0;
  int tid;
  int teams = 2;
  
  // omp_set_num_threads(2);
  
  
  #pragma omp parallel private(tid)
  {
    
    #pragma omp master
    {
    
      printf("master start\n");
      for (int i = 3; i <= 16; i++) {
        next = 0;
        sleep(3);
        teams++;
        next = 1;
        // sleep(1);
        printf("master step to: %d\n", i);
      }
      
      quit = 1;
      printf("master end\n");
    }
    
    
    while(!quit) {
      while(next);
      //#pragma omp parallel for
      for (int j = 0; j < teams; j++) {
        tid = omp_get_thread_num();
        printf("%d teams. work on: %d. this is tid %d\n", teams, j, tid);
        while (!next);
        printf("tid: %d, finished work, there are: %d\n", tid, omp_get_num_threads());
      }     
    }
  }
}