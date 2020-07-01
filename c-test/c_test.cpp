#include<string>
#include<iostream>
using namespace std;
void quickSort(string str, int left, int right){
    int i=left;
    int j=right;
    char tmp=str[i];
    if(i<j){
      while(i < j){
        while(i < j and str[j] >= tmp )
            --j;
        if(i<j){
            str[i]=str[j];
            ++i;
            cout << str[i] << "\t";
            cout << tmp << "\n";
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
void countStr(const string str){
    int len=str.size();
    if (len==0)
        return ;
    cout << str << "\n";
    quickSort(str, 0, len-1);
    int j=1;
    cout << str << "\n";
    for(int i=1; i<len; i++){
        if(str[i]==str[i-1]){
            i++;
            j++;
        }
        else {
            cout << str[i-1] << j;
        }
    }
}

int main(){
    string str;
    cin >> str;
    cout << str.size() << "\n";
    countStr(str);
    cout << "\n";
}
