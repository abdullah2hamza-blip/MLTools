// Research driver: stock ActionABI CPU library APIs, explicit synthetic inputs.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "actionabi/converter.hpp"
#include "actionabi/evidence.hpp"
#include "actionabi/identifiability.hpp"
#include "actionabi/probe.hpp"
#include "actionabi/score.hpp"
#include "actionabi/trajectory.hpp"
#include <nlohmann/json.hpp>

using nlohmann::json;
using namespace actionabi;

std::string status_name(FieldStatus value) {
  switch (value) {
  case FieldStatus::Identified:
    return "identified";
  case FieldStatus::Ambiguous:
    return "ambiguous";
  case FieldStatus::Unsupported:
    return "unsupported";
  }
  throw std::invalid_argument("unknown field status");
}

json encode_equivalence(const EquivalenceReport &report) {
  json fields = json::object(), retained = json::array();
  for (const auto &[field, value] : report.fields)
    fields[field] = status_name(value);
  for (const auto &value : report.retained)
    retained.push_back(to_json(value.contract));
  return {{"fields", fields},
          {"retained", retained},
          {"caller_supplied_calibrated_coverage", report.calibrated_coverage}};
}

json run(const json &request) {
  const auto operation = request.at("operation").get<std::string>();
  if (operation == "score") {
    const auto dataset = load_jsonl(request.at("dataset").get<std::string>());
    ScoreOptions options{request.at("huber_delta"),
                         request.at("train_episode_fraction")};
    std::vector<ScoredHypothesis> scored;
    for (const auto &encoded : request.at("hypotheses")) {
      const auto contract = contract_from_json(encoded);
      scored.push_back({contract, score_cpu(dataset, contract, options)});
    }
    const auto unsupported =
        request.at("unsupported_fields").get<std::set<std::string>>();
    const auto equivalence =
        analyze_identifiability(scored, request.at("score_threshold"),
                                request.at("coverage_input"), unsupported);
    EvidenceReport evidence{
        .schema_version = "1.0",
        .dataset_sha256 = request.at("dataset_sha256"),
        .source_filename = request.at("dataset"),
        .train_episode_ids = {0},
        .heldout_episode_ids = {1},
        .score_options = options,
        .ranked = scored,
        .counterexamples = {},
        .warnings = {"Synthetic two-episode development control; no empirical "
                     "coverage claim"},
        .backend = "cpu",
        .build_revision = "5ad1d45a082860f5f16bb43f3c6bbb0758ec6c11",
    };
    return {{"native_evidence", evidence_to_json(evidence)},
            {"equivalence", encode_equivalence(equivalence)}};
  }
  if (operation == "convert") {
    std::vector<ScoredHypothesis> scored;
    for (const auto &row : request.at("score_evidence")
                               .at("native_evidence")
                               .at("ranked_hypotheses")) {
      const auto &score = row.at("score");
      scored.push_back(
          {contract_from_json(row.at("contract")),
           {score.at("train_loss"), score.at("heldout_loss"),
            score.at("residual_count"),
            score.at("per_dimension_loss").get<std::vector<double>>()}});
    }
    const auto report = analyze_identifiability(
        scored, request.at("score_threshold"), request.at("coverage_input"),
        request.at("unsupported_fields").get<std::set<std::string>>());
    const auto evidence_hash = request.at("evidence_sha256").get<std::string>();
    try {
      const auto converter = make_converter(report, evidence_hash);
      const auto encoded = converter_to_json(converter);
      const auto loaded = converter_from_json(encoded, evidence_hash);
      bool bad_hash_rejected = false;
      try {
        static_cast<void>(converter_from_json(encoded, std::string(64, '0')));
      } catch (const std::invalid_argument &) {
        bad_hash_rejected = true;
      }
      return {{"decision", "EMIT"},
              {"converter", encoded},
              {"converted",
               convert_instantaneous(
                   loaded, request.at("action").get<std::vector<double>>())},
              {"bad_evidence_hash_rejected", bad_hash_rejected}};
    } catch (const std::invalid_argument &error) {
      return {{"decision", "REFUSE"}, {"reason", error.what()}};
    }
  }
  if (operation == "pulses") {
    return {
        {"pulses", bounded_axis_pulses(request.at("lower"), request.at("upper"),
                                       request.at("amplitude"))}};
  }
  if (operation == "select") {
    std::vector<ProbeCandidate> candidates;
    for (const auto &row : request.at("candidates")) {
      candidates.push_back(
          {row.at("action"), row.at("predicted_observations"), row.at("safe")});
    }
    const auto selected = select_separating_probes(
        candidates, request.at("observation_tolerance"));
    json pairs = json::array();
    for (const auto &pair : selected.inseparable_pairs)
      pairs.push_back({pair.first, pair.second});
    return {{"selected_candidate_indices", selected.selected_candidate_indices},
            {"inseparable_pairs", pairs},
            {"exact", selected.exact}};
  }
  if (operation == "calibration") {
    const auto scores =
        request.at("calibration_scores").get<std::vector<double>>();
    bool insufficient = false;
    try {
      static_cast<void>(split_conformal_threshold(scores, 0.1));
    } catch (const InsufficientCalibrationError &) {
      insufficient = true;
    }
    return {{"threshold", split_conformal_threshold(scores, 0.25)},
            {"insufficient_calibration_rejected", insufficient}};
  }
  throw std::invalid_argument("unsupported research driver operation");
}

int main(int argc, char **argv) {
  try {
    if (argc != 2 || std::filesystem::file_size(argv[1]) > 2 * 1024 * 1024)
      throw std::invalid_argument("one bounded JSON request is required");
    std::ifstream input(argv[1]);
    const auto request = json::parse(input);
    std::cout << run(request).dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << json{{"error", error.what()}}.dump() << '\n';
    return 2;
  }
}
