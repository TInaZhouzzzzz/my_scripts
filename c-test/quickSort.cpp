#include<string>
#include<iostream>
#include<random>
using namespace std;
using std::default_random_engine;

void quickSort(int str[], int left, int right){
    int i=left;
    int j=right;
    int tmp=str[i];
    if(i<j){
      while(i < j) {
        while(i < j and str[j] >= tmp )
            --j;
        if(i<j){
            str[i]=str[j];
            ++i;
        }
        while(i<j and str[i] < tmp)
            ++i;
        if(i<j){
            str[j]=str[i];
            --j;
        }
      }
      str[i]=tmp;
      quickSort(str, left, i-1);
      quickSort(str, i+1, right);
    }
}

int main(){
  default_random_engine e;
  int a[12];
  int b[12];

  quickSort(a, 0, 11);
  for(int i=0; i<12; i++) {
    cout << a[i] << "\t";
  }
  cout << "\n";
  quickSort(b, 0, 11);
  for(int i=0; i<12; i++) {
    cout << b[i] << "\t";
  }
  cout << "\n";
}
