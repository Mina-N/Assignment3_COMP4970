

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <random>

using namespace std;

#define NUM_DATA_POINTS 595
#define NUM_FEATURES 95


class Data_Point {
    public:
    int pnt_id; // Identification number
    double clsfr; // Classifier
    double feat_vecs[NUM_FEATURES]; // Array for each of the 95 features of each element
    
    /* Constructor takes in a line from the dataset, saves the id, classifier,
     and saves each of the features into the vector. */
    Data_Point() {  // Instantiation
        pnt_id = -1;
    }
    
    Data_Point(string line) { // Initialization
        istringstream iss;
        iss.str(line);
        
        iss >> pnt_id;
        iss >> clsfr;
        
        int i;
        for(i = 0; i < NUM_FEATURES; i++)
        iss >> feat_vecs[i];
    }
    
    
    // Classifies Data_Point using vector of length
    double grnn_classify(Data_Point trng_set[], double* sigma, int pop_index);
    
    double grnn_classify(Data_Point trng_set[], double sigma);
    
    double grnn_classify(Data_Point trng_set[], double sigma, double* weights, int pop_index);
    
    double knn_classify(Data_Point trng_set[], int k, double b);
};


class Features {
public:
    int pop_size;
    double* fitness;
    double* weights;
    int* rank;
    
    Features(int pop_size_in) {
        pop_size = pop_size_in;
        
        fitness = new double[pop_size];
        weights = new double[NUM_FEATURES*pop_size];
        rank = new int[pop_size];
    }
    
    Features(Data_Point trng_set[], int pop_size_in) {
        // Dynamically allocate instance variables
        pop_size = pop_size_in;
        fitness = new double[pop_size];
        weights = new double[NUM_FEATURES*pop_size];
        rank = new int[pop_size];
        
        // Max and min values for sigma
        double min_weight = 0;
        double max_weight = 1;
        
        // Declare temporary instance variables
        double* temp_fitness;
        temp_fitness = new double[pop_size];
        double* temp_weights;
        temp_weights = new double[NUM_FEATURES*pop_size];
        int* temp_rank;
        temp_rank = new int[pop_size];
        
        // Seed rand()
        srand(time(0));
        
        // Initialize population
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < NUM_FEATURES; j++) {
                temp_weights[i*NUM_FEATURES + j] = ((double) rand() / RAND_MAX) * (max_weight - min_weight) + min_weight;
                cout << "Random weight: " << temp_weights[i*NUM_FEATURES + j] << endl;
            }
        }
        
        // Initialize fitness to 0
        for (int i = 0; i < pop_size; i++) {
            temp_fitness[i] = 0;
        }
        double prediction;
        int clsfr_rate;
        
        // Evaluate the fitness of initial population
        cout << "\nInitial Fitness of Population" << endl;
        for (int i = 0; i < pop_size; i++) {
            clsfr_rate = 0;
            prediction = 0;
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
                prediction = trng_set[j].grnn_classify(trng_set, 0.1237, temp_weights, i);
                if (i > 1) {
                    //cout << "Prediction: " << prediction << endl;
                }
                // Assess prediction
                if (prediction < 0 && trng_set[j].clsfr < 0 || prediction > 0 && trng_set[j].clsfr > 0) {
                    clsfr_rate++;
                }
                //cout << "temp_fitness[i]: " << temp_fitness[i] << endl;
            }
            //cout << "temp_fitness[i]: " << temp_fitness[i] << endl;
            temp_fitness[i] = ((double) clsfr_rate) / NUM_DATA_POINTS;
            // Print fitness of individuals
            cout << "Individual " << i << ": " << temp_fitness[i] << endl;
            
            // Rank individuals in population for sorting and for linear rank proportional parent selection
            temp_rank[i] = 1;
            for (int j = 0; j < i; j++) {
                if (temp_fitness[i] > temp_fitness[j]) {
                    temp_rank[j]++;
                }
                else {
                    temp_rank[i]++;
                }
            }
        }
        
        // Sort parents based on rank
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < NUM_FEATURES; j++) {
                weights[NUM_FEATURES*(temp_rank[i]-1) + j] = temp_weights[NUM_FEATURES*i + j];
            }
            fitness[temp_rank[i]-1] = temp_fitness[i];
            rank[temp_rank[i]-1] = temp_rank[i];
        }
        
        // Free dynamically allocated memory
        delete [] temp_fitness;
        delete [] temp_weights;
        delete [] temp_rank;
    }
    
    void assess_fitness(Data_Point trng_set[], int child) {
        
        // Initialize and allocate memory for temporary variables
        double* temp_fitness;
        temp_fitness = new double[pop_size];
        double* temp_weights;
        temp_weights = new double[pop_size*NUM_FEATURES];
        int* temp_rank;
        temp_rank = new int[pop_size];
        double prediction;
        
        int clsfr_rate;
        
        // Evaluate fitness of population
        for (int i = 0; i < pop_size; i++) {
            prediction = 0;
            clsfr_rate = 0;
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
                prediction = trng_set[j].grnn_classify(trng_set, 0.1237, weights, i);
                if (prediction < 0 && trng_set[j].clsfr < 0 || prediction > 0 && trng_set[j].clsfr > 0) {
                    clsfr_rate++;
                }
                if (clsfr_rate > NUM_DATA_POINTS) {
                    cout << "temp_fitness overflowing: " << temp_fitness[i] << endl;
                }
                //cout << "temp_fitness[i]: " << temp_fitness[i] << endl;
            }
            //cout << "temo_fitness[i]: " << temp_fitness[i] << endl;
            temp_fitness[i] = ((double) clsfr_rate) / NUM_DATA_POINTS;
            temp_rank[i] = 1;
            for (int j = 0; j < i; j++) {
                if (temp_fitness[i] > temp_fitness[j]) {
                    temp_rank[j]++;
                }
                else {
                    temp_rank[i]++;
                }
            }
        }
        
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < NUM_FEATURES; j++) {
                temp_weights[NUM_FEATURES*i + j] = weights[NUM_FEATURES*i + j];
            }
        }
        
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < NUM_FEATURES; j++) {
                weights[NUM_FEATURES*(temp_rank[i]-1) + j] = temp_weights[NUM_FEATURES*i + j];
            }
            fitness[temp_rank[i]-1] = temp_fitness[i];
            rank[temp_rank[i]-1] = temp_rank[i];
        }
        if (child == 1) {
            cout << "\nFitness of Children" << endl;
            for (int i = 0; i < pop_size; i++) {
                cout << "Child " << i << ": " << fitness[i] << endl;
            }
        }
        
        delete [] temp_weights;
        delete [] temp_fitness;
        delete [] temp_rank;
    }
    
    void select_parents(void) {
        /**
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
            }
        }
         */
        Features temp_parents(pop_size);
        srand(time(0));
        int P;
        cout << "\nFitness of Parents" << endl;
        for (int i = 0; i < pop_size; i++) {
            cout << "Parent " << i << ": " << fitness[i] << endl;
        }
        for (int i = 0; i < pop_size; i++) {
            P = (rand() % (pop_size * (pop_size + 1)/2)) + 1;
            for (int j = pop_size; j > 0; j--) {
                P -= j;
                if (P <=0) {
                    temp_parents.fitness[i] = fitness[pop_size-j];
                    for (int k = 0; k < NUM_FEATURES; k++) {
                        temp_parents.weights[NUM_FEATURES*i + k] = weights[NUM_FEATURES*(pop_size-j) + k];
                    }
                    temp_parents.rank[i] = rank[pop_size-j];
                    break;
                }
            }
        }
        
        for (int i = 0; i < pop_size; i++) {
            fitness[i] = temp_parents.fitness[i];
            for (int j = 0; j < NUM_FEATURES; j++) {
                weights[NUM_FEATURES*i + j] = temp_parents.weights[NUM_FEATURES*i + j];
            }
            rank[i] = temp_parents.rank[i];
        }
        cout << "\nFitness of Selected Parents" << endl;
        for (int i = 0; i < pop_size; i++) {
            cout << "Selected Parent " << i << ": " << fitness[i] << endl;
        }
        /**
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
            }
        }
         */
    }
    
    double procreate(Data_Point trng_set[], int k) {
        Features temp_children(pop_size);
        Features k_best_parents(k);
        
        // Mutation rate
        double mutation_dev = 0.15;
        default_random_engine generator(time(0));
        normal_distribution<double> mutation(0.0,1.0);
        
        if (pop_size % 2 == 0) {
            double parent1_weight;
            double parent2_weight;
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < NUM_FEATURES; j++) {
                    temp_children.weights[i*NUM_FEATURES + j] = weights[i*NUM_FEATURES + j] + mutation_dev * mutation(generator);
                }
            }
            for (int i = k; i < pop_size; i += 2) {
                for (int j = 0; j < NUM_FEATURES; j++) {
                    parent1_weight = weights[NUM_FEATURES*i + j];
                    parent2_weight = weights[NUM_FEATURES*(i+1) + j];
                    if (parent1_weight > parent2_weight) {
                        temp_children.weights[NUM_FEATURES*i + j] = ((double) rand() / RAND_MAX) * (parent1_weight - parent2_weight) + parent2_weight + mutation_dev * mutation(generator);
                        temp_children.weights[NUM_FEATURES*(i+1) + j] = ((double) rand() / RAND_MAX) * (parent1_weight - parent2_weight) + parent2_weight + mutation_dev * mutation(generator);
                    }
                    else {
                        temp_children.weights[NUM_FEATURES*i + j] = ((double) rand() / RAND_MAX) * (parent2_weight - parent1_weight) + parent1_weight + mutation_dev * mutation(generator);
                        temp_children.weights[NUM_FEATURES*(i+1) + j] = ((double) rand() / RAND_MAX) * (parent2_weight - parent1_weight) + parent1_weight + mutation_dev * mutation(generator);
                    }
                }
            }
        }
        
        else {
            double parent1_weight;
            double parent2_weight;
            for (int i = 0; i < k+1; i++) {
                for (int j = 0; j < NUM_FEATURES; j++) {
                    temp_children.weights[i*false + j] = weights[i*NUM_FEATURES + j] + mutation_dev * mutation(generator);
                }
            }
            for (int i = k+1; i < pop_size - k; i +=2) {
                for (int j = 0; j < NUM_FEATURES; j++) {
                    parent1_weight = weights[NUM_FEATURES*i + j];
                    parent2_weight = weights[NUM_FEATURES*(i+1) + j];
                    if (parent1_weight > parent2_weight) {
                        temp_children.weights[NUM_FEATURES*i + j] = ((double) rand() / RAND_MAX) * (parent1_weight - parent2_weight) + parent2_weight + mutation_dev * mutation(generator);
                        temp_children.weights[NUM_FEATURES*(i+1) + j] = ((double) rand() / RAND_MAX) * (parent1_weight - parent2_weight) + parent2_weight + mutation_dev * mutation(generator);
                    }
                    else {
                        temp_children.weights[NUM_FEATURES*i + j] = ((double) rand() / RAND_MAX) * (parent2_weight - parent1_weight) + parent1_weight + mutation_dev * mutation(generator);
                        temp_children.weights[NUM_FEATURES*(i+1) + j] = ((double) rand() / RAND_MAX) * (parent2_weight - parent1_weight) + parent1_weight + mutation_dev * mutation(generator);
                    }
                }
            }
        }
        temp_children.assess_fitness(trng_set, 1);
        
        assess_fitness(trng_set, 0);
        
        for (int i = k; i < pop_size; i++) {
            for (int j = 0; j < NUM_FEATURES; j++) {
                weights[NUM_FEATURES*i + j] = temp_children.weights[NUM_FEATURES*(i-k) + j];
            }
            fitness[i] = temp_children.fitness[i-k];
            rank[i] = temp_children.rank[i-k];
        }
        
        assess_fitness(trng_set, 0);
        return fitness[0];
    }
    
    ~Features() {
        delete [] fitness;
        delete [] weights;
        delete [] rank;
    }
    
};



class Std_Deviation {
    public:
    int pop_size;
    double* fitness;
    double* sigma;
    int* rank;
    
    // Dummy Constructor
    Std_Deviation(int pop_size_in) {
        pop_size = pop_size_in;
 
        fitness = new double[pop_size];
        sigma = new double[NUM_DATA_POINTS*pop_size];
        rank = new int[pop_size];
    }
    
    
    // Main Constructor - Initilize population of size pop_size_in to random values between 0 and 2
    Std_Deviation(Data_Point trng_set[], int pop_size_in) {
        
        // Dynamically allocate instance variables
        pop_size = pop_size_in;
        fitness = new double[pop_size];
        sigma = new double[NUM_DATA_POINTS*pop_size];
        rank = new int[pop_size];
        
        // Max and min values for sigma
        double min_sigma = 0.00001;
        double max_sigma = 0.25;
        
        // Declare temporary instance variables
        double* temp_fitness;
        temp_fitness = new double[pop_size];
        double* temp_sigma;
        temp_sigma = new double[NUM_DATA_POINTS*pop_size];
        int* temp_rank;
        temp_rank = new int[pop_size];
        
        // Seed rand()
        srand(time(0));
        
        // Initialize population
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
                temp_sigma[i*NUM_DATA_POINTS + j] = ((double) rand() / RAND_MAX) * (max_sigma - min_sigma) + min_sigma;
            }
        }
        
        // Initialize fitness to 0
        for (int i = 0; i < pop_size; i++) {
            temp_fitness[i] = 0;
        }
        double prediction;
        
        // Evaluate the fitness of initial population
        cout << "\nInitial Fitness of Population" << endl;
        for (int i = 0; i < pop_size; i++) {
            prediction = 0;
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
                prediction = trng_set[j].grnn_classify(trng_set, temp_sigma, i);
                // Assess prediction
                if (prediction < 0 && trng_set[j].clsfr < 0 || prediction > 0 && trng_set[j].clsfr > 0) {
                    temp_fitness[i]++;
                }
            }
            temp_fitness[i] = temp_fitness[i] / NUM_DATA_POINTS;
            // Print fitness of individuals
            cout << "Individual " << i << ": " << temp_fitness[i] << endl;
            
            // Rank individuals in population for sorting and for linear rank proportional parent selection
            temp_rank[i] = 1;
            for (int j = 0; j < i; j++) {
                if (temp_fitness[i] > temp_fitness[j]) {
                    temp_rank[j]++;
                }
                else {
                    temp_rank[i]++;
                }
            }
        }
        
        // Sort parents based on rank
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
                sigma[NUM_DATA_POINTS*(temp_rank[i]-1) + j] = temp_sigma[NUM_DATA_POINTS*i + j];
            }
            fitness[temp_rank[i]-1] = temp_fitness[i];
            rank[temp_rank[i]-1] = temp_rank[i];
        }
        
        // Free dynamically allocated memory
        delete [] temp_fitness;
        delete [] temp_sigma;
        delete [] temp_rank;
        
    }
    
    // return fitness of individual at specified index
    void set_fitness(double fitness_in, int index) {
        fitness[index] = fitness_in;
    }
    
    // Assess the fitness of rank population
    void assess_fitness(Data_Point trng_set[], int child) {
        
        // Initialize and allocate memory for temporary variables
        double* temp_fitness;
        temp_fitness = new double[pop_size];
        double* temp_sigma;
        temp_sigma = new double[pop_size*NUM_DATA_POINTS];
        int* temp_rank;
        temp_rank = new int[pop_size];
        double prediction;
        
        // Evaluate fitness of population
        for (int i = 0; i < pop_size; i++) {
            prediction = 0;
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
                prediction = trng_set[j].grnn_classify(trng_set, sigma, i);
                if (prediction < 0 && trng_set[j].clsfr < 0 || prediction > 0 && trng_set[j].clsfr > 0) {
                    temp_fitness[i]++;
                }
                if (temp_fitness[i] > NUM_DATA_POINTS) {
                    cout << "temp_fitness overflowing: " << temp_fitness[i] << endl;
                }
            }
            temp_fitness[i] /= NUM_DATA_POINTS;
            temp_rank[i] = 1;
            for (int j = 0; j < i; j++) {
                if (temp_fitness[i] > temp_fitness[j]) {
                    temp_rank[j]++;
                }
                else {
                    temp_rank[i]++;
                }
            }
        }
        
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
                temp_sigma[NUM_DATA_POINTS*i + j] = sigma[NUM_DATA_POINTS*i + j];
            }
        }
        
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
                sigma[NUM_DATA_POINTS*(temp_rank[i]-1) + j] = temp_sigma[NUM_DATA_POINTS*i + j];
            }
            fitness[temp_rank[i]-1] = temp_fitness[i];
            rank[temp_rank[i]-1] = temp_rank[i];
        }
        if (child == 1) {
            cout << "\nFitness of Children" << endl;
            for (int i = 0; i < pop_size; i++) {
                cout << "Child " << i << ": " << fitness[i] << endl;
            }
        }
        
        delete [] temp_sigma;
        delete [] temp_fitness;
        delete [] temp_rank;
    }
    
    void select_parents(void) {
        /**
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
            }
        }
         */
        Std_Deviation temp_parents(pop_size);
        srand(time(0));
        int P;
        cout << "\nFitness of Parents" << endl;
        for (int i = 0; i < pop_size; i++) {
            cout << "Parent " << i << ": " << fitness[i] << endl;
        }
        for (int i = 0; i < pop_size; i++) {
            P = (rand() % (pop_size * (pop_size + 1)/2)) + 1;
            for (int j = pop_size; j > 0; j--) {
                P -= j;
                if (P <=0) {
                    temp_parents.fitness[i] = fitness[pop_size-j];
                    for (int k = 0; k < NUM_DATA_POINTS; k++) {
                        temp_parents.sigma[NUM_DATA_POINTS*i + k] = sigma[NUM_DATA_POINTS*(pop_size-j) + k];
                    }
                    temp_parents.rank[i] = rank[pop_size-j];
                    break;
                }
            }
        }
        
        for (int i = 0; i < pop_size; i++) {
            fitness[i] = temp_parents.fitness[i];
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
                sigma[NUM_DATA_POINTS*i + j] = temp_parents.sigma[NUM_DATA_POINTS*i + j];
            }
            rank[i] = temp_parents.rank[i];
        }
        cout << "\nFitness of Selected Parents" << endl;
        for (int i = 0; i < pop_size; i++) {
            cout << "Selected Parent " << i << ": " << fitness[i] << endl;
        }
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
            }
        }
    }
    
    double procreate(Data_Point trng_set[], int k) {
        Std_Deviation temp_children(pop_size);
        Std_Deviation k_best_parents(k);
        
        // Mutation rate
        double mutation_dev = 0.005;
        default_random_engine generator(time(0));
        normal_distribution<double> mutation(0.0,1.0);
        
        if (pop_size % 2 == 0) {
            double parent1_sigma;
            double parent2_sigma;
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < NUM_DATA_POINTS; j++) {
                    temp_children.sigma[i*NUM_DATA_POINTS + j] = sigma[i*NUM_DATA_POINTS + j] + mutation_dev * mutation(generator);
                }
            }
            for (int i = k; i < pop_size; i += 2) {
                for (int j = 0; j < NUM_DATA_POINTS; j++) {
                    parent1_sigma = sigma[NUM_DATA_POINTS*i + j];
                    parent2_sigma = sigma[NUM_DATA_POINTS*(i+1) + j];
                    if (parent1_sigma > parent2_sigma) {
                        temp_children.sigma[NUM_DATA_POINTS*i + j] = ((double) rand() / RAND_MAX) * (parent1_sigma - parent2_sigma) + parent2_sigma + mutation_dev * mutation(generator);
                        temp_children.sigma[NUM_DATA_POINTS*(i+1) + j] = ((double) rand() / RAND_MAX) * (parent1_sigma - parent2_sigma) + parent2_sigma + mutation_dev * mutation(generator);
                    }
                    else {
                        temp_children.sigma[NUM_DATA_POINTS*i + j] = ((double) rand() / RAND_MAX) * (parent2_sigma - parent1_sigma) + parent1_sigma + mutation_dev * mutation(generator);
                        temp_children.sigma[NUM_DATA_POINTS*(i+1) + j] = ((double) rand() / RAND_MAX) * (parent2_sigma - parent1_sigma) + parent1_sigma + mutation_dev * mutation(generator);
                    }
                }
            }
        }
        
        else {
            double parent1_sigma;
            double parent2_sigma;
            for (int i = 0; i < k+1; i++) {
                for (int j = 0; j < NUM_DATA_POINTS; j++) {
                    temp_children.sigma[i*NUM_DATA_POINTS + j] = sigma[i*NUM_DATA_POINTS + j] + mutation_dev * mutation(generator);
                }
            }
            for (int i = k+1; i < pop_size - k; i +=2) {
                for (int j = 0; j < NUM_DATA_POINTS; j++) {
                    parent1_sigma = sigma[NUM_DATA_POINTS*i + j];
                    parent2_sigma = sigma[NUM_DATA_POINTS*(i+1) + j];
                    if (parent1_sigma > parent2_sigma) {
                        temp_children.sigma[NUM_DATA_POINTS*i + j] = ((double) rand() / RAND_MAX) * (parent1_sigma - parent2_sigma) + parent2_sigma + mutation_dev * mutation(generator);
                        temp_children.sigma[NUM_DATA_POINTS*(i+1) + j] = ((double) rand() / RAND_MAX) * (parent1_sigma - parent2_sigma) + parent2_sigma + mutation_dev * mutation(generator);
                    }
                    else {
                        temp_children.sigma[NUM_DATA_POINTS*i + j] = ((double) rand() / RAND_MAX) * (parent2_sigma - parent1_sigma) + parent1_sigma + mutation_dev * mutation(generator);
                        temp_children.sigma[NUM_DATA_POINTS*(i+1) + j] = ((double) rand() / RAND_MAX) * (parent2_sigma - parent1_sigma) + parent1_sigma + mutation_dev * mutation(generator);
                    }
                }
            }
        }
        temp_children.assess_fitness(trng_set, 1);
        
        assess_fitness(trng_set, 0);
        
        for (int i = k; i < pop_size; i++) {
            for (int j = 0; j < NUM_DATA_POINTS; j++) {
                sigma[NUM_DATA_POINTS*i + j] = temp_children.sigma[NUM_DATA_POINTS*(i-k) + j];
            }
            fitness[i] = temp_children.fitness[i-k];
            rank[i] = temp_children.rank[i-k];
        }
        
        assess_fitness(trng_set, 0);
        return fitness[0];
    }
    
    ~Std_Deviation() {
        delete [] fitness;
        delete [] sigma;
        delete [] rank;
    }

};

void init_trng_set(Data_Point trng_set[]);

int main() {
    
    // Empirically determined optimal uniform standard deviation value for Gaussian kernel
    double sigma = 0.1237;
    
    Data_Point trng_set[NUM_DATA_POINTS];
    
    init_trng_set(trng_set);
    
    cout << "\n***Generation 0***" << endl;
    Features features(trng_set, 20);
    
    
    double best_fitness = 0;
    double current_fitness;
    for (int i = 0; i < 10000; i++) {
        features.select_parents();
        current_fitness = features.procreate(trng_set, 2);
        if (current_fitness > best_fitness) {
            best_fitness = current_fitness;
            cout << "Current Best Weights:" << endl;
            for (int i = 0; i < NUM_FEATURES; i++) {
                cout << "Weight " << i << ": " << features.weights[i] << endl;
            }
        }
        cout << "\nBest Fitness: " << best_fitness;
        cout << "\n***Generation " << i+1 << "***\n" << endl;
    }
    
    
    int TP, TN, FP, FN, U;
    double grnn_clsf_rate;
    grnn_clsf_rate = 0;
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    U = 0;
    for (int i = 0; i < NUM_DATA_POINTS; i++) {
        double prediction = trng_set[i].grnn_classify(trng_set, sigma);
        if (prediction < 0 && trng_set[i].clsfr < 0) {
            TN++;
            grnn_clsf_rate++;
        }
        else if (prediction > 0 && trng_set[i].clsfr > 0) {
            TP++;
            grnn_clsf_rate++;
        }
        else if (prediction < 0 && trng_set[i].clsfr > 0) {
            FN++;
        }
        else if (prediction > 0 && trng_set[i].clsfr < 0) {
            FP++;
        }
        else {
            U++;
        }
    }
    grnn_clsf_rate /= NUM_DATA_POINTS;
    cout << "Prediction Rate: " << grnn_clsf_rate << endl;
    cout << "True Positives: " << TP << endl;
    cout << "True Negatives: " << TN << endl;
    cout << "False Positives: " << FP << endl;
    cout << "False Negatives: " << FN << endl;
    if (U > 0) {
        cout << "Unclassifiables: " << U << endl;
    }
    /**
    cout << "\nDistance weighted k-nearest neighbors" << endl;
    double knn_clsf_rate;
    int k = 3;
    double b = 3.1;
    while (b < 4.3) {
        knn_clsf_rate = 0;
        for (int i = 0; i < NUM_DATA_POINTS; i++) {
            double prediction = trng_set[i].knn_classify(trng_set, k, b);
            if (prediction < 0 && trng_set[i].clsfr < 0 || prediction > 0 && trng_set[i].clsfr > 0) {
                knn_clsf_rate++;
            }
        }
        knn_clsf_rate /= NUM_DATA_POINTS;
        
        cout << "b = " << b << endl;
        cout << "Prediction Rate: " << knn_clsf_rate << endl;
        b += 0.01;
    }
     */
    
    return 0;
}

void init_trng_set(Data_Point trng_set[]) {
    ifstream input("our_dataset.txt");
    string line;
    int j = 0;
    double magnitude;
    while (getline(input, line)) {
        trng_set[j++] = Data_Point(line);
        magnitude = 0;
        for (int k = 0; k < NUM_FEATURES; k++)
            magnitude += pow(trng_set[j-1].feat_vecs[k], 2);
        if (magnitude != 0) {
            for (int k = 0; k < NUM_FEATURES; k++)
                trng_set[j-1].feat_vecs[k] /= sqrt(magnitude);
        }
    }
}

double Data_Point::grnn_classify(Data_Point trng_set[], double* sigma, int pop_index) {
    double gaussian = 0;
    double weighted_gaussian = 0;
    for (int i = 0; i < NUM_DATA_POINTS; i++) {
        double distance_squared = 0;
        for (int j = 0; j < NUM_FEATURES; j++) {
            distance_squared += pow((trng_set[i].feat_vecs[j] - this->feat_vecs[j]),2);
        }
        if (distance_squared != 0) {
            gaussian += exp(-distance_squared/(2*pow(sigma[NUM_DATA_POINTS*pop_index + i],2)));
            weighted_gaussian += exp(-distance_squared/(2*pow(sigma[NUM_DATA_POINTS*pop_index + i],2))) * trng_set[i].clsfr;
        }
    }
    return weighted_gaussian / gaussian;
}

double Data_Point::grnn_classify(Data_Point trng_set[], double sigma) {
    double gaussian = 0;
    double weighted_gaussian = 0;
    for (int i = 0; i < NUM_DATA_POINTS; i++) {
        double distance_squared = 0;
        for (int j = 0; j < NUM_FEATURES; j++) {
            distance_squared += pow((trng_set[i].feat_vecs[j] - this->feat_vecs[j]),2);
        }
        if (distance_squared != 0) {
            gaussian += exp(-distance_squared/(2*pow(sigma,2)));
            weighted_gaussian += exp(-distance_squared/(2*pow(sigma,2))) * trng_set[i].clsfr;
        }
    }
    return weighted_gaussian / gaussian;
}

double Data_Point::grnn_classify(Data_Point trng_set[], double sigma, double* weights, int pop_index) {
    double gaussian = 0;
    double weighted_gaussian = 0;
    for (int i = 0; i < NUM_DATA_POINTS; i++) {
        double distance_squared = 0;
        for (int j = 0; j < NUM_FEATURES; j++) {
            distance_squared += weights[NUM_FEATURES*pop_index + j] * pow((trng_set[i].feat_vecs[j] - this->feat_vecs[j]),2);
        }
        if (distance_squared != 0) {
            gaussian += exp(-distance_squared/(2*pow(sigma,2)));
            weighted_gaussian += exp(-distance_squared/(2*pow(sigma,2))) * trng_set[i].clsfr;
        }
    }
    return weighted_gaussian / gaussian;
}

double Data_Point::knn_classify(Data_Point trng_set[], int k, double b) {
    double* k_clsfr;
    k_clsfr = new double[k];
    double* k_distances;
    k_distances = new double[k];
    double min_distance = 80000;
    int min_k_index;
    double distance;
    for (int i = 0; i < k; i++) {
        min_distance = 80000;
        for (int j = 0; j < NUM_DATA_POINTS; j++) {
            distance = 0;
            for (int l = 0; l < NUM_FEATURES; l++) {
                distance += pow((trng_set[j].feat_vecs[l] - this->feat_vecs[l]), 2);
            }
            //cout << "distance: " << distance << endl;
            bool in_array = false;
            for (int m = 0; m < i; m++) {
                if (fabs(distance - k_distances[m]) < 0.0000001) {
                    in_array = true;
                    break;
                }
            }
            if (distance < min_distance && !in_array && distance > 0.0000001) {
                min_distance = distance;
                min_k_index = j;
            }

        }
        //cout << "k_distances[i-1] " << k_distances[i-1] << endl;
        //cout << "min_distance " << min_distance << endl;
        //cout << "fabs(distance - k_distances[i-1]): " << fabs(min_distance - k_distances[i-1]) << endl;
        k_distances[i] = min_distance;
        //cout << min_k_index << endl;
        k_clsfr[i] = trng_set[min_k_index].clsfr;
        //cout << k_distances[i] << endl;
    }
    double prediction = 0;
    double denominator = 0;
    for (int i = 0; i < k; i++) {
        prediction += k_clsfr[i] / pow(k_distances[i], b);
        denominator += 1 / pow(k_distances[i], b);
    }
    prediction /= denominator;
    delete k_clsfr;
    delete k_distances;
    return prediction;
}






