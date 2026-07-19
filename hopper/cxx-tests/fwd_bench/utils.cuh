#pragma once

#include <string>
#include <vector>

#include <csv2/writer.hpp>
#include <argparse/argparse.hpp>

void write_result_to_csv(
  std::string const& filename,
  std::vector<std::string> const& header,
  std::vector<std::vector<std::string>> const& data
){
  std::ofstream csv_file(filename);
  csv2::Writer<csv2::delimiter<','>> writer(csv_file);
  writer.write_row(header);
  for (const auto& row : data) {
    writer.write_row(row);
  }
  csv_file.close();
}

void add_write_result_to_csv(  // 追加
  std::string const& filename,
  std::vector<std::string> const& header,
  std::vector<std::vector<std::string>> const& data
){
  std::ofstream csv_file(filename, std::ios::app);
  csv2::Writer<csv2::delimiter<','>> writer(csv_file);

  if (csv_file.tellp() == 0) {
    writer.write_row(header);
  }

  for (const auto& row : data) {
    writer.write_row(row);
  }
}

std::string parse_filename_arg(
  int argc, char* argv[]
){
  argparse::ArgumentParser program("bench");
  program.add_argument("filename").default_value("").help("CSV filename to save the results");
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    exit(1);
  }
  std::string filename = program.get<std::string>("filename");
  return filename;
}