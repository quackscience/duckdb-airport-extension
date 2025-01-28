#pragma once

#include "airport_extension.hpp"
#include "duckdb.hpp"

#include "duckdb/function/table/arrow.hpp"
#include <msgpack.hpp>

namespace duckdb
{
  unique_ptr<ArrowArrayStreamWrapper> AirportProduceArrowScan(const ArrowScanFunctionData &function,
                                                              const vector<column_t> &column_ids,
                                                              TableFilterSet *filters);

  void AirportTakeFlight(ClientContext &context, TableFunctionInput &data_p, DataChunk &output);

  unique_ptr<GlobalTableFunctionState> AirportArrowScanInitGlobal(ClientContext &context,
                                                                  TableFunctionInitInput &input);

  struct AirportTakeFlightParameters
  {
    string server_location;
    string auth_token;
    string secret_name;
    // Override the ticket supplied from GetFlightInfo.
    // this is supplied via a named parameter.
    string ticket;

    std::unordered_map<string, std::vector<string>> user_supplied_headers;
  };

  AirportTakeFlightParameters AirportParseTakeFlightParameters(
      string &server_location,
      ClientContext &context,
      TableFunctionBindInput &input);

  struct GetFlightInfoTableFunctionParameters
  {
    std::string schema_name;
    std::string action_name;
    std::string parameters;
    MSGPACK_DEFINE_MAP(schema_name, action_name, parameters)
  };

  unique_ptr<FunctionData> AirportTakeFlightBindWithFlightDescriptor(
      const AirportTakeFlightParameters &take_flight_params,
      const arrow::flight::FlightDescriptor &descriptor,
      ClientContext &context,
      TableFunctionBindInput &input,
      vector<LogicalType> &return_types,
      vector<string> &names,
      std::shared_ptr<arrow::flight::FlightInfo> *cached_info_ptr,
      std::shared_ptr<GetFlightInfoTableFunctionParameters> table_function_parameters);

}
