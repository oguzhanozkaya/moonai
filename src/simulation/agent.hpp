#pragma once

#include "core/types.hpp"
#include "evolution/genome.hpp"
#include "evolution/neural_network.hpp"

#include <memory>

namespace moonai {

enum class AgentType { Predator, Prey };

class Agent {
public:
  Agent(AgentId id, AgentType type, Vec2 position, float speed,
        float vision_range, float energy);
  virtual ~Agent() = default;

  virtual void update(float dt) = 0;

  // Apply a movement decision from the neural network (dx, dy normalized)
  void apply_movement(Vec2 direction, float dt);

  AgentId id() const {
    return id_;
  }
  AgentType type() const {
    return type_;
  }
  Vec2 position() const {
    return position_;
  }
  Vec2 velocity() const {
    return velocity_;
  }
  float speed() const {
    return speed_;
  }
  float vision_range() const {
    return vision_range_;
  }
  float energy() const {
    return energy_;
  }
  int age() const {
    return age_;
  }
  bool alive() const {
    return alive_;
  }
  float fitness() const {
    return fitness_;
  }
  int kills() const {
    return kills_;
  }
  int food_eaten() const {
    return food_eaten_;
  }
  float distance_traveled() const {
    return distance_traveled_;
  }
  int species_id() const {
    return species_id_;
  }
  int offspring_count() const {
    return offspring_count_;
  }
  int reproduction_cooldown() const {
    return reproduction_cooldown_;
  }
  const Genome &genome() const {
    return genome_;
  }
  Genome &genome() {
    return genome_;
  }
  NeuralNetwork *network() const {
    return network_.get();
  }
  NeuralNetwork *network() {
    return network_.get();
  }

  void set_position(Vec2 pos) {
    position_ = pos;
  }
  void set_velocity(Vec2 vel) {
    velocity_ = vel;
  }
  void set_energy(float energy) {
    energy_ = energy;
  }
  void set_age(int age) {
    age_ = age;
  }
  void set_distance_traveled(float distance) {
    distance_traveled_ = distance;
  }
  void set_kills(int kills) {
    kills_ = kills;
  }
  void set_food_eaten(int food_eaten) {
    food_eaten_ = food_eaten;
  }
  void set_species_id(int id) {
    species_id_ = id;
  }
  void set_alive(bool alive) {
    alive_ = alive;
  }
  void set_offspring_count(int count) {
    offspring_count_ = count;
  }
  void set_reproduction_cooldown(int cooldown) {
    reproduction_cooldown_ = cooldown;
  }
  void set_genome(Genome genome, const std::string &activation_function) {
    genome_ = std::move(genome);
    rebuild_network(activation_function);
  }
  void rebuild_network(const std::string &activation_function) {
    network_ = std::make_unique<NeuralNetwork>(genome_, activation_function);
  }
  void add_fitness(float amount) {
    fitness_ += amount;
  }
  void add_energy(float amount) {
    energy_ += amount;
  }
  void drain_energy(float amount) {
    energy_ -= amount;
  }
  void increment_age() {
    ++age_;
    if (reproduction_cooldown_ > 0) {
      --reproduction_cooldown_;
    }
  }
  void add_kill() {
    ++kills_;
  }
  void add_food() {
    ++food_eaten_;
  }
  void add_offspring() {
    ++offspring_count_;
  }

  bool is_dead() const {
    return energy_ <= 0.0f;
  }

protected:
  AgentId id_;
  AgentType type_;
  Vec2 position_;
  Vec2 velocity_ = {0.0f, 0.0f};
  float speed_;
  float vision_range_;
  float energy_;
  int age_ = 0;
  bool alive_ = true;
  float fitness_ = 0.0f;
  int kills_ = 0;
  int food_eaten_ = 0;
  float distance_traveled_ = 0.0f;
  int species_id_ = -1;
  int offspring_count_ = 0;
  int reproduction_cooldown_ = 0;
  Genome genome_;
  std::unique_ptr<NeuralNetwork> network_;
};

} // namespace moonai
