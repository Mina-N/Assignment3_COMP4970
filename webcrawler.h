#ifndef WEBCRAWLER_H
#define WEBCRAWLER_H

#include <iostream>
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

using namespace std;

int find_children(string filename, int depth, stack<string>& stack1, stack<int>& stack2, int i, bool leaf);
void char_extractor(string filename, int k);
void webcrawler(stack<string>& url_stack, stack<int>& depth_stack, int depth_limit, int& i);

#endif
