#include <stdio.h>
#include <iostream>
#include <vector>
#include <queue>
#include "find_top_k.h"
using namespace std;
 
void swap(FT &a, FT &b) {
  a = a + b;
  b = a - b;
  a = a - b;
}
 
// to swap the indices
void swap(int& a, int& b) {
    a = a + b;
    b = a - b;
    a = a - b;
}
 
void minHeapify(FT a[], int size, int i, size_t indices[]) {
  int l = 2* i ;
  int r = 2* i + 1;
  int smallest = i;
  if (l < size && a[l] < a[smallest])
    smallest = l;
  if (r < size && a[r] < a[smallest])
    smallest = r;
  if (smallest != i) {
    swap(a[i], a[smallest]);
    swap(indices[i], indices[smallest]);
    minHeapify(a, size, smallest, indices);
  }
 
}
 
void buildMinHeap(FT a[], int size, size_t indices[]) {
  int i;
  for (i = size / 2; i >= 0; i--)
    minHeapify(a, size, i, indices);
}
 
void kthLargest(FT a[], int size, int k, size_t indices[]) {
  FT minHeap[k];
  int i;
  for (i = 0; i < k; i++) {
    minHeap[i] = a[i];
    indices[i] = i;
  }
  buildMinHeap(minHeap, k, indices);
  for (i = k; i < size; i++) {
    if (a[i] > minHeap[0]) {
      minHeap[0] = a[i];
      indices[0] = i;
      minHeapify(minHeap, k, 0, indices);
    }
  }
}
 
// int main() {
//     int a[] = { 916, 17, 2666, 4, 12, 9, 5, 100 };
//     int size = sizeof(a) / sizeof(a[0]);
//     int k = 5;
//     //printf("\n%d ",kthLargest(a,size,k));
//     int ind[k];
//     int i;
//     kthLargest(a, size, k, ind);
//     for (i = 0; i < k; ++i)
//         printf("%d ", ind[i]);
//     std::cout << "\n";
 
//     // priority queue to test our results
//     std::vector<int> test = { 916, 17, 2666, 4, 12, 9, 5, 100 };
//     std::priority_queue<std::pair<int, int> > q;
//     int min = test[0]; // this should be a limit
//     for(int i = 0; i < k; ++i) {
//         q.push(std::pair<int, int>(test[i], i));
//         if(test[i] < min)
//             min = test[i];
//     }
//     for (int i =k; i < (int) test.size(); ++i) {
//         if(min <= test[i]) {
//             // if you don't add things that are smaller than the smallest item
//             //already on the queue.
//             q.push(std::pair<int, int>(test[i], i));
//         }
//     }
//     for (int i = 0; i < k; ++i) {
//         int ki = q.top().second;
//         std::cout << "index[" << i << "] = " << ki << std::endl;
//         q.pop();
//     }
 
//     return 0;
// }