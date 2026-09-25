#include "core/_module.hpp"
#include "parser/parser_light.hpp"
#include "decPOMDP/sequential/oSarsa.hpp"
#include <string>

using namespace common;

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : "../example/epidemic_light.dpomdp";
    const std::string log = argc > 2 ? argv[2] : "day_one.csv";

    // Configure the sequential planner before Support::init(), because the
    // history representation allocates memory from SEARCH.truncation.
    SEARCH.seed = 0;
    SEARCH.horizon = 2;
    SEARCH.timeout = 2.0;
    SEARCH.log_filename = log;
    SEARCH.verbose = utils::Verbose::none;
    SEARCH.algo_name = "oSarsa-seq";
    SEARCH.compress = false;
    SEARCH.truncation = 3;
    SEARCH.compress_threshold = 1e-8;
    SEARCH.epsilon_start = .5f;
    SEARCH.use_reward_shaping = false;
    SEARCH.use_portfolio = true;
    SEARCH.use_simulatedAnnealing = true;
    SEARCH.workers = 1;
    SEARCH.lazy_reachability = false;

    parser_light::Parser parser(model);
    parser.encode_problem();
    PROBLEM.bench_name = model;
    decPOMDP::Support::init();

    decPOMDP::sequential::oSarsa().solve();
    return 0;
}
