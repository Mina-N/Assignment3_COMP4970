# Assignment3_COMP4970

To run main_driver.cpp, which contains the updated web crawler code from Assignment 1, place the main_driver.cpp, webcrawler.cpp, getWebPage.class, webcrawler.h, and grnn.cpp files in the same directory. Create a new directory on the same level as the previous four files named html_source. Created another new directory on the same level as html_source called unigrams. To compile and link the files, execute

 g++ -c webcrawler.cpp
 
 g++ -c main_driver.cpp
 
 g++ -o main_driver webcrawler.cpp main_driver.cpp
 
 To compile grnn.cpp, execute
 
 g++ -o grnn grnn.cpp
  
To run main_driver.cpp, execute

./main_driver

During execution, the webcrawler() function in main_driver.cpp adds a feature vector corresponding to an expanded URL to an output file in the directory unigrams. Our Vulnerability Analyzer reads in, analyzes, and produces a classification for the feature vector. This process will repeat until forty classifications have been produced: 10 classifications in the range -1.0 to -0.5, 10 classifications in the range -0.5 to 0.0, 10 classifications in the range 0.0 to 0.5, and 10 classifications in the range 0.5 to 1.0. 

Then, these forty new classifications are added to our original dataset, and the accuracy of our Vulnerability Analyzer is determined and compared to its baseline accuracy (that is, its accuracy without using the forty new classifications in the dataset). 
