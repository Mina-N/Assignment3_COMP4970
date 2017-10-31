//
// webcrawler.cpp
//  webcrawler
//
//  Created by Mina Narayanan on 10/28/17.
//  Copyright © 2017 Mina Narayanan. All rights reserved.
//

#include "webcrawler.h"


//using namespace std;

// Finds the children of filename and stores them and their associated depths into stacks. Writes the html of each expanded
// URL to a .txt file.
//int find_children(string filename, int depth, stack<string>& stack1, stack<int>& stack2, int i, bool leaf);

// Takes text file path as keyboard input and writes the unigram vector to new text file. Writes relative
// frequency of both printable characters.
//void char_extractor(string filename, int k);

//void webcrawler();

/*
int main() {

    string filepath;
    int depth_limit = -1;
    int depth = 0;
    int i = 0;

    //stack that stores URLs
    stack<string> url_stack;

    //stack that stores depths of URLs
    stack<int> depth_stack;

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

    //IMPROVEMENT: WE ARE USING DEPTH-LIMITED SEARCH INSTEAD OF ITERATIVE DEEPENING SEARCH
    while (!url_stack.empty()) {

        //pop url from queue
        filepath = url_stack.top();
        cout << "filepath being expanded is " << filepath << endl;
        url_stack.pop();


        //set depth to depth associated with url by checking array --> necessary for backtracking up the tree of URLs
        depth = depth_stack.top();
        depth_stack.pop();
        cout << "the depth of the filepath being expanded is " << depth << endl;


        //Find children of URL and store them into the stack. Also store their associated depths into an stack. Saves HTML of expanded URL in a .txt file
        find_children(filepath, depth, url_stack, depth_stack, i, depth >= depth_limit);

        char_extractor("output_" + to_string(i) + ".txt", i);
        i++;
    }

    return 0;

}
*/

void char_extractor(string filename, int k) {
    char c;
    double unigram_count[95] = {0};
    double total_unigrams = 0;
    string feature_vectors = "feature_vectors.txt";

    ifstream in_text("html_source/" + filename);

    while (in_text >> noskipws >> c) {
        if (isprint(c)) {
            int i = c - 32;
            unigram_count[i]++;
            total_unigrams++;
        }
    }

    in_text.close();

    // Create and write to new text file containing relative frequencies

    // string new_filename = filename.erase(filename.length() - 4, 4) + "_unigrams.txt";
    // ofstream out_text("unigrams/" + new_filename);

    ofstream out_text(feature_vectors, ofstream::app);
    out_text << k;

    for (int i = 0; i < 95; i++) {
        //char a = i + 32;
        // IMPROVEMENT: NORMALIZE UNIGRAM FREQUENCY BY TOTAL UNIGRAMS
        // out_text << a << ": " << unigram_count[i] / total_unigrams << "\n";
        out_text << " " << (unigram_count[i] / total_unigrams);
    }
    out_text << "\n";

    out_text.close();
}



int find_children(string filename_str, int depth, stack<string>& stack1, stack<int>& stack2, int i, bool leaf) {

    char javaCall[100] = "";
    char url[100] = "";
    string new_url_str, unnecessary = "";
    int num_of_links = 0;
    string j = to_string(i);
    char *k = new char[j.size()+1];
    strcpy(k, j.c_str());

    char *filename = new char[filename_str.size()+1];
    strcpy(filename, filename_str.c_str());

    // Build call to getWebPage

    strcat(javaCall, "java getWebPage ");
    strcat(javaCall, filename);
    strcat(javaCall, " > html_source/output_");
    strcat(javaCall, k);
    strcat(javaCall, ".txt");
    system(javaCall);
    ifstream webpage("html_source/output_" + j + ".txt");

    while (!webpage.eof() && !leaf) {
        webpage >> setw(99) >> url;
        if (strncmp("href", url, 4) == 0) {
            // cout << url << endl;
            string url_str(url);
            //TODO: FIGURE OUT IF URL HAS NOT ALREADY BEEN SEEN --> FIX IF THERE IS TIME

            //Remove href=" from the front of the URL and remove " from the end of the URL:
            //Use quotation marks as delimiters

            stringstream ss(url_str);


            getline(ss, unnecessary, '"');
            getline(ss, new_url_str, '"');

            //IMPROVEMENT: prevents any URLS that do not begin with "http" from being pushed onto the stack

            if ((new_url_str[0] != '#') && (new_url_str.substr(new_url_str.length() - 3, new_url_str.length()) != "pdf") && (new_url_str.substr(0, 4) == "http") && strncmp(url, filename, sizeof(url)/sizeof(url[0]))) {
                //push URL onto stack
                stack1.push(new_url_str);
                cout << "Stack1: " << stack1.top() << endl;

                //store associated depth of URL into stack as (depth + 1)
                stack2.push(depth + 1);
                cout << "Stack2: " << stack2.top() << endl;
                num_of_links++;
            }
        }
    }

  //  cout << "depth " << (i + 1) << " done" << endl;

    webpage.close();

    return num_of_links;
}

void webcrawler(stack<string>& url_stack, stack<int>& depth_stack, int depth_limit, int& i) {

  string filepath = "";
  int depth = 0;
  /*
  int i = 0;
  int depth_limit = -1;

  //stack that stores URLs
  stack<string> url_stack;

  //stack that stores depths of URLs
  stack<int> depth_stack;

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
*/

  //IMPROVEMENT: WE ARE USING DEPTH-LIMITED SEARCH INSTEAD OF ITERATIVE DEEPENING SEARCH
//  while (!url_stack.empty()) {

      //pop url from queue
      filepath = url_stack.top();
      cout << "filepath being expanded is " << filepath << endl;
      url_stack.pop();


      //set depth to depth associated with url by checking array --> necessary for backtracking up the tree of URLs
      depth = depth_stack.top();
      depth_stack.pop();
      cout << "the depth of the filepath being expanded is " << depth << endl;


      //Find children of URL and store them into the stack. Also store their associated depths into a stack. Saves HTML of expanded URL in a .txt file
      find_children(filepath, depth, url_stack, depth_stack, i, depth >= depth_limit);

      char_extractor("output_" + to_string(i) + ".txt", i);
      i++;
  //  }
}
