#pragma once

#include "assert.h"
#include <vector>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <functional>
#include <bitset>
#include <initializer_list>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>


//-- define MARKTIME and TIME as a simple way to compute elapsed time
inline std::chrono::high_resolution_clock::time_point now;
#define MARKTIME now = std::chrono::high_resolution_clock::now();
#define TIME std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - now).count()
inline std::chrono::high_resolution_clock::time_point _now;
#define _MARKTIME _now = std::chrono::high_resolution_clock::now();
#define _TIME std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - _now).count()

namespace utils{    
    using namespace std;

    int get_jointIndex(const vector<int>& , const vector<int>& , int );
    vector<int> get_indivIndices(const vector<int>& max_indices, int agents_number, int jointIndex);

    //-- ANSI
    enum Colors:int{
        black=30, red, green, yellow, blue, magenta, cyan, white, // foreground
        black_b=40, red_b, green_b, yellow_b, blue_b, magenta_b, cyan_b, white_b, // background
        reset=0, bold=1, underline=4, inverse=7, bold_off=21, underline_off=24, inverse_off=27 // inverse = swap foreground and background colours
        };
    string _get_colorStr(int color);
    string _get_colorStr(initializer_list<int> color_list);
    void _print_color(int color, string msg);
    void _print_color(initializer_list<int> color_list, string msg);
    string get_solver_repr(string solver_name);
    void print_red(string msg);
    void print_green(string msg);
    void print_yellow(string msg);
    void print_magenta(string msg);
    void print_cyan(string msg);
    

    //-- product
    template <typename T>
    T get_product(const initializer_list<T> & list){
        return accumulate(list.begin(), list.end(), 1, std::multiplies<T>());
    }

    template <typename T>
    T get_product(const vector<T> & list){
        return accumulate(list.begin(), list.end(), 1, std::multiplies<T>());
    }

    //-- kronecker
    template<typename T>
    inline double kronecker(T i, T j){
        return i == j ? 1 : 0;
    }

    //-- hash
    template <class T>
    inline void hash_combine(size_t & s, const T & v)
    {
        hash<T> h;
        s^= h(v) + 0x9e3779b9 + (s<< 6) + (s>> 2);
    }

    template <class T1, class T2>
    struct Pair_Hash{// : unary_function<pair<T1, T2>, size_t> {
        size_t operator()(pair<T1, T2> const& val) const {      
            size_t seed = 0;
            hash_combine(seed, val.first);
            hash_combine(seed, val.second);            
            return seed;
        }
    };

    //-- binary

    /// @brief Print the bits of a number.
    template<typename T>
    string get_binary_repr(T x){
        ostringstream os;
        os << "0b" << (bitset<8*sizeof(T)>(x));
        return os.str();
    }
    
    //-- seed
    void seed_init(int seed = -1, bool verbose = true);

    //-- fast calc
    namespace fast_calc{
        inline unsigned int g_seed;
        ///@brief Seed the generator
        inline void seed_init(int seed) {
            g_seed = seed;
        }
        ///@brief returns one integer, similar output value range as C lib.
        inline int _rand() {
            g_seed = (214013*g_seed+2531011);
            return (g_seed>>16)&0x7FFF;
        }

        inline int fastRandRange(int maxSize) { // return an integer from [0;maxSize[
            return _rand() % maxSize;
        }


        ///@brief return an integer from [0;maxSize[
        inline int rand_range(int maxSize) {
            return _rand() % maxSize;
        }
        ///@brief return an integer from [a;b]
        inline int rand_int(int a, int b) {
            return(a + rand_range(b - a + 1));
        }
        ///@brief return a number in [0,1]
        template<typename T = double>
        inline T rand() {
            return (static_cast<T>(_rand()) / 0x7FFF);
        }
        ///@brief return a double in [a, b]
        inline double rand_double(double a, double b) {
            return a + (static_cast<double>(_rand()) / 0x7FFF)*(b-a);
        }
        ///@brief return a size_t
        template<typename T>
        inline T rand_hash(){
            static const int nb_bits = sizeof(T)*8;
            T val = 0;
            T one = 1;
            for (int b=0; b<nb_bits; b++){
                if (rand_double(0, 1) > 0.5)
                    val |= (one<<b);
            }
            return val;
        }
        static inline double log(const double& x){
            union{double f;uint32_t i;} vx = { x };
            double y = vx.i;y *= 8.2629582881927490e-8f;
            return(y - 87.989971088f);
        }
        static inline double sqrt(const double& x){
            union{int i;double x;}u;u.x=x;u.i=(1<<29)+(u.i>>1)-(1<<22);
            return(u.x);
        }
    }

    //////////////////
    // Logger
    //////////////////
    enum Verbose: int {none = 0, low = 1, medium = 2, high = 3};

    class Logger{ 
        fstream file;
        string filename;
        Verbose verbose;
        bool append_mode;
        double last_write_time;
    public:

        Logger(string filename, Verbose verbose, bool append_mode = false);
        ~Logger();

        void write(const initializer_list<string> & list, double time = 0);
        double get_time_sinceLastWrite(double current_time);        
    };
}
