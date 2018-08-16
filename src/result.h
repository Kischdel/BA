#ifndef RESULT_H
#define RESULT_H

typedef struct resultFGMRES {
  int restart, iter, steps;
  double time, timeJacobi, averageTimeJacobi;
} resultFGMRES;


#endif //RESULT_H