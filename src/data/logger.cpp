#include "data/logger.hpp"

#include "data/metrics.hpp"
#include "evolution/genome.hpp"
#include "evolution/species.hpp"
#include "simulation/agent.hpp"
#include "simulation/simulation_manager.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <sstream>

namespace moonai {

namespace {

const char *event_name(SimEvent::Type type) {
  switch (type) {
    case SimEvent::Kill:
      return "kill";
    case SimEvent::Food:
      return "food";
    case SimEvent::Birth:
      return "birth";
    case SimEvent::Death:
      return "death";
  }
  return "unknown";
}

} // namespace

Logger::Logger(const std::string &output_dir, std::uint64_t seed,
               const std::string &name)
    : base_dir_(output_dir), name_(name), seed_(seed) {}

Logger::~Logger() {
  flush_steps();
  if (events_file_.is_open() && !events_buffer_.empty()) {
    events_file_ << events_buffer_;
  }
  if (genomes_file_.is_open()) {
    genomes_file_ << "\n]";
    genomes_file_.close();
  }
  if (stats_file_.is_open()) {
    stats_file_.close();
  }
  if (species_file_.is_open()) {
    species_file_.close();
  }
  if (steps_file_.is_open()) {
    steps_file_.close();
  }
  if (events_file_.is_open()) {
    events_file_.close();
  }
}

bool Logger::initialize(const SimulationConfig &config) {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &time);
#else
  localtime_r(&time, &tm);
#endif

  std::string dir_name;
  if (!name_.empty()) {
    dir_name = name_;
  } else {
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S") << "_seed" << seed_;
    dir_name = oss.str();
  }
  run_dir_ = base_dir_ + "/" + dir_name;

  if (std::filesystem::exists(run_dir_)) {
    for (int suffix = 2; suffix < 1000; ++suffix) {
      std::string candidate =
          base_dir_ + "/" + dir_name + "_" + std::to_string(suffix);
      if (!std::filesystem::exists(candidate)) {
        run_dir_ = candidate;
        break;
      }
    }
  }

  std::filesystem::create_directories(run_dir_);
  save_config(config, run_dir_ + "/config.json");

  stats_file_.open(run_dir_ + "/stats.csv");
  if (!stats_file_.is_open()) {
    return false;
  }
  stats_file_ << "# master_seed=" << seed_ << "\n";
  stats_file_ << "step,predator_count,prey_count,births,deaths,best_fitness,"
                 "avg_fitness,num_species,avg_complexity,avg_predator_energy,"
                 "avg_prey_energy\n";

  species_file_.open(run_dir_ + "/species.csv");
  if (species_file_.is_open()) {
    species_file_ << "# master_seed=" << seed_ << "\n";
    species_file_ << "step,species_id,size,avg_fitness,best_fitness\n";
  }

  genomes_file_.open(run_dir_ + "/genomes.json");
  if (!genomes_file_.is_open()) {
    return false;
  }
  genomes_file_ << "[";

  if (config.step_log_enabled) {
    steps_file_.open(run_dir_ + "/steps.csv");
    events_file_.open(run_dir_ + "/events.csv");
    if (!steps_file_.is_open() || !events_file_.is_open()) {
      return false;
    }
    steps_file_ << "step,agent_id,type,alive,x,y,energy,kills,food_eaten,"
                   "offspring_count,species_id\n";
    events_file_
        << "step,event_type,agent_id,target_id,parent_a_id,parent_b_id,x,y\n";
  }

  return true;
}

void Logger::log_report(const StepMetrics &metrics) {
  if (!stats_file_.is_open()) {
    return;
  }
  stats_file_ << metrics.step << ',' << metrics.predator_count << ','
              << metrics.prey_count << ',' << metrics.births << ','
              << metrics.deaths << ',' << metrics.best_fitness << ','
              << metrics.avg_fitness << ',' << metrics.num_species << ','
              << metrics.avg_genome_complexity << ','
              << metrics.avg_predator_energy << ',' << metrics.avg_prey_energy
              << '\n';
}

void Logger::log_best_genome(int step, const Genome &genome) {
  if (!genomes_file_.is_open()) {
    return;
  }

  nlohmann::json j;
  j["step"] = step;
  j["fitness"] = genome.fitness();
  j["num_nodes"] = genome.nodes().size();
  j["num_connections"] = genome.connections().size();
  j["genome"] = nlohmann::json::parse(genome.to_json());

  if (!genomes_first_entry_) {
    genomes_file_ << ',';
  }
  genomes_file_ << '\n' << j.dump(2);
  genomes_first_entry_ = false;
}

void Logger::log_species(int step, const std::vector<Species> &species) {
  if (!species_file_.is_open()) {
    return;
  }
  for (const auto &entry : species) {
    species_file_ << step << ',' << entry.id() << ',' << entry.members().size()
                  << ',' << entry.average_fitness() << ','
                  << entry.best_fitness_ever() << '\n';
  }
}

void Logger::log_step(int step,
                      const std::vector<std::unique_ptr<Agent>> &agents) {
  if (!steps_file_.is_open()) {
    return;
  }
  for (const auto &agent : agents) {
    steps_buffer_ +=
        std::to_string(step) + "," + std::to_string(agent->id()) + "," +
        (agent->type() == AgentType::Predator ? "predator" : "prey") + "," +
        (agent->alive() ? "1" : "0") + "," +
        std::to_string(agent->position().x) + "," +
        std::to_string(agent->position().y) + "," +
        std::to_string(agent->energy()) + "," + std::to_string(agent->kills()) +
        "," + std::to_string(agent->food_eaten()) + "," +
        std::to_string(agent->offspring_count()) + "," +
        std::to_string(agent->species_id()) + "\n";
    ++steps_buffered_;
  }
  if (steps_buffered_ >= STEP_FLUSH_EVERY) {
    flush_steps();
  }
}

void Logger::log_events(int step, const std::vector<SimEvent> &events) {
  if (!events_file_.is_open() || events.empty()) {
    return;
  }
  for (const auto &event : events) {
    events_buffer_ += std::to_string(step) + "," + event_name(event.type) +
                      "," + std::to_string(event.agent_id) + "," +
                      std::to_string(event.target_id) + "," +
                      std::to_string(event.parent_a_id) + "," +
                      std::to_string(event.parent_b_id) + "," +
                      std::to_string(event.position.x) + "," +
                      std::to_string(event.position.y) + "\n";
    ++events_buffered_;
  }
  if (events_buffered_ >= STEP_FLUSH_EVERY) {
    events_file_ << events_buffer_;
    events_buffer_.clear();
    events_buffered_ = 0;
  }
}

void Logger::flush_steps() {
  if (!steps_file_.is_open() || steps_buffer_.empty()) {
    return;
  }
  steps_file_ << steps_buffer_;
  steps_buffer_.clear();
  steps_buffered_ = 0;
}

void Logger::flush() {
  if (stats_file_.is_open()) {
    stats_file_.flush();
  }
  if (genomes_file_.is_open()) {
    genomes_file_.flush();
  }
  if (species_file_.is_open()) {
    species_file_.flush();
  }
  flush_steps();
  if (events_file_.is_open() && !events_buffer_.empty()) {
    events_file_ << events_buffer_;
    events_buffer_.clear();
    events_buffered_ = 0;
  }
  if (steps_file_.is_open()) {
    steps_file_.flush();
  }
  if (events_file_.is_open()) {
    events_file_.flush();
  }
}

} // namespace moonai
