/*#include <iostream>
#include <string>
#include <ctype.h>
#include <fstream>
#include <cstring>
#include <stdlib.h>
#include <iomanip>
#include <stack>
#include <stdio.h>
#include <cstdlib>
#include <sstream>
*/
#include "webcrawler.h"


//using namespace std;

int range1 = 0; // -1.0 to -0.5
int range2 = 0; // -0.5 to 0.0
int range3 = 0; // 0.0 to 0.5
int range4 = 0; // 0.5 to 1.0

//stack that stores URLs
stack<string> url_stack;
//stack that stores depths of URLs
stack<int> depth_stack;

string filepath;
int depth_limit = -1;
int depth = 0;
int i = 0;


int main() {

  //prompting input
  cout << "Please input a root URL: ";
  cin >> filepath;

  //checking user input
  while (depth_limit < 0) {
      cout << "Please input a depth limit >= 0:  ";
      cin >> depth_limit;
  }

  //push root filepath onto stack
  url_stack.push(filepath);

  //push root depth onto stack
  depth_stack.push(0);

  //stopping conditions
  while (range1 < 10 || range2 < 10 || range3 < 10 || range4 < 10) {

    //exhausted searching for URLs to the depth limit
    if (url_stack.empty()) {
      break;
    }
    //Provide webcrawler with a URL, which prints out
    //feature vectors to a file one by one
    webcrawler(url_stack, depth_stack, depth_limit, i);

    //Have Matt's stuff read in the current feature vector and classify it, updating one of the range variables

  }

return 0;

}
