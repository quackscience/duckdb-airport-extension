#include "duckdb.hpp"
#include "storage/airport_delete.hpp"
#include "storage/airport_table_entry.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "storage/airport_catalog.hpp"
#include "storage/airport_transaction.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/common/arrow/arrow_appender.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "airport_macros.hpp"
#include "airport_headers.hpp"
#include "airport_exception.hpp"
#include "airport_secrets.hpp"

#include "arrow/array/array_dict.h"
#include "arrow/array/array_nested.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/buffer.h"
#include "arrow/io/memory.h"
#include "arrow/ipc/options.h"
#include "arrow/ipc/reader.h"
#include "arrow/ipc/type_fwd.h"
#include "arrow/ipc/writer.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/c/bridge.h"
#include "duckdb/common/arrow/schema_metadata.hpp"

#include "airport_flight_stream.hpp"
#include "airport_take_flight.hpp"
#include "storage/airport_exchange.hpp"
#include "yyjson.hpp"

using namespace duckdb_yyjson; // NOLINT

namespace duckdb
{
  static std::string join_vector_of_strings(const std::vector<std::string> &vec, const char joiner)
  {
    if (vec.empty())
      return "";

    return std::accumulate(
        std::next(vec.begin()), vec.end(), vec.front(),
        [joiner](const std::string &a, const std::string &b)
        {
          return a + joiner + b;
        });
  }

  static int findIndex(const std::vector<std::string> &vec, const std::string &target)
  {
    auto it = std::find(vec.begin(), vec.end(), target);

    if (it == vec.end())
    {
      throw std::runtime_error("String not found in vector");
    }

    return std::distance(vec.begin(), it);
  }

  void AirportExchangeGetGlobalSinkState(ClientContext &context,
                                         const TableCatalogEntry &table,
                                         const AirportTableEntry &airport_table,
                                         AirportExchangeGlobalState *global_state,
                                         ArrowSchema &send_schema,
                                         bool return_chunk,
                                         string exchange_operation,
                                         vector<string> destination_chunk_column_names)
  {
    global_state->schema = arrow::ImportSchema(&send_schema).ValueOrDie();
    global_state->flight_descriptor = airport_table.table_data->flight_info->descriptor();

    auto auth_token = AirportAuthTokenForLocation(context, airport_table.table_data->location, "", "");

    D_ASSERT(airport_table.table_data != nullptr);

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto flight_location, flight::Location::Parse(airport_table.table_data->location),
                                                       airport_table.table_data->location,
                                                       global_state->flight_descriptor, "");

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto flight_client,
                                                       flight::FlightClient::Connect(flight_location),
                                                       airport_table.table_data->location,
                                                       global_state->flight_descriptor, "");

    auto trace_uuid = UUID::ToString(UUID::GenerateRandomUUID());

    arrow::flight::FlightCallOptions call_options;
    airport_add_standard_headers(call_options, airport_table.table_data->location);

    if (!auth_token.empty())
    {
      std::stringstream ss;
      ss << "Bearer " << auth_token;
      call_options.headers.emplace_back("authorization", ss.str());
    }

    call_options.headers.emplace_back("airport-trace-id", trace_uuid);

    // Indicate that we are doing a delete.
    call_options.headers.emplace_back("airport-operation", exchange_operation);

    // Indicate if the caller is interested in data being returned.
    call_options.headers.emplace_back("return-chunks", return_chunk ? "1" : "0");

    if (global_state->flight_descriptor.type == arrow::flight::FlightDescriptor::PATH)
    {
      auto path_parts = global_state->flight_descriptor.path;
      std::string joined_path_parts = join_vector_of_strings(path_parts, '/');
      call_options.headers.emplace_back("airport-flight-path", joined_path_parts);
    }

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(
        auto exchange_result,
        flight_client->DoExchange(call_options, global_state->flight_descriptor),
        airport_table.table_data->location,
        global_state->flight_descriptor, "");

    // Tell the server the schema that we will be using to write data.
    AIRPORT_ARROW_ASSERT_OK_LOCATION_DESCRIPTOR(
        exchange_result.writer->Begin(global_state->schema),
        airport_table.table_data->location,
        global_state->flight_descriptor,
        "Begin schema");

    // Now that there is a reader stream and a writer stream, we want to reuse the Arrow
    // scan code as much as possible, but the problem is it assumes its being called as
    // part of a table returning function, with the life cycle of bind, init global, init local
    // and scan.
    //
    // But we can simulate most of that here.
    auto scan_data = make_uniq<AirportTakeFlightScanData>(
        airport_table.table_data->location,
        airport_table.table_data->flight_info,
        //        std::move(flight_info),
        std::move(exchange_result.reader));

    auto scan_bind_data = make_uniq<AirportExchangeTakeFlightBindData>(
        (stream_factory_produce_t)&AirportFlightStreamReader::CreateStream,
        (uintptr_t)scan_data.get());

    scan_bind_data->scan_data = std::move(scan_data);
    scan_bind_data->flight_client = std::move(flight_client);
    scan_bind_data->server_location = airport_table.table_data->location;
    scan_bind_data->trace_id = trace_uuid;

    vector<column_t> column_ids;

    if (return_chunk)
    {
      AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto read_schema,
                                                         scan_bind_data->scan_data->stream_->GetSchema(),
                                                         airport_table.table_data->location,
                                                         global_state->flight_descriptor, "");

      // printf("Schema of reader stream is:\n----------\n%s\n---------\n", read_schema->ToString().c_str());

      auto &data = *scan_bind_data;
      AIRPORT_ARROW_ASSERT_OK_LOCATION_DESCRIPTOR(
          ExportSchema(*read_schema, &data.schema_root.arrow_schema),
          airport_table.table_data->location,
          global_state->flight_descriptor,
          "ExportSchema");

      vector<string> reading_arrow_column_names;
      for (idx_t col_idx = 0;
           col_idx < (idx_t)data.schema_root.arrow_schema.n_children; col_idx++)
      {
        auto &schema = *data.schema_root.arrow_schema.children[col_idx];
        if (!schema.release)
        {
          throw InvalidInputException("airport_exchange: released schema passed");
        }
        auto name = string(schema.name);
        if (name.empty())
        {
          name = string("v") + to_string(col_idx);
        }

        reading_arrow_column_names.push_back(name);
      }

      // printf("Arrow schema column names are: %s\n", join_vector_of_strings(reading_arrow_column_names, ',').c_str());
      // printf("Expected order of columns to be: %s\n", join_vector_of_strings(destination_chunk_column_names, ',').c_str());

      vector<string> arrow_types;
      for (idx_t col_idx = 0;
           col_idx < (idx_t)data.schema_root.arrow_schema.n_children; col_idx++)
      {
        auto &schema = *data.schema_root.arrow_schema.children[col_idx];
        if (!schema.release)
        {
          throw InvalidInputException("airport_exchange: released schema passed");
        }
        auto arrow_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), schema);
        arrow_types.push_back(arrow_type->GetDuckType().ToString());
      }

      for (auto output_index = 0; output_index < destination_chunk_column_names.size(); output_index++)
      {
        auto found_index = findIndex(reading_arrow_column_names, destination_chunk_column_names[output_index]);
        if (exchange_operation != "update")
        {
          column_ids.push_back(found_index);
        }
        else
        {
          // This is right for outputs, because it allowed the read chunk to happen.
          column_ids.push_back(output_index);
        }
        // printf("Output data chunk column %s (type=%s) (%d) comes from arrow column index index %d\n",
        //        destination_chunk_column_names[output_index].c_str(),
        //        arrow_types[found_index].c_str(),
        //        output_index,
        //        found_index);
      }

      // idx_t row_id_col_idx = -1;
      for (idx_t col_idx = 0;
           col_idx < (idx_t)data.schema_root.arrow_schema.n_children; col_idx++)
      {
        auto &schema = *data.schema_root.arrow_schema.children[col_idx];
        if (!schema.release)
        {
          throw InvalidInputException("airport_exchange: released schema passed");
        }
        auto arrow_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), schema);

        // Determine if the column is the row_id column by looking at the metadata
        // on the column.
        bool is_row_id_column = false;
        if (schema.metadata != nullptr)
        {
          auto column_metadata = ArrowSchemaMetadata(schema.metadata);

          auto comment = column_metadata.GetOption("is_row_id");
          if (!comment.empty())
          {
            is_row_id_column = true;
            scan_bind_data->row_id_column_index = col_idx;
          }
        }

        if (schema.dictionary)
        {
          auto dictionary_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), *schema.dictionary);
          arrow_type->SetDictionary(std::move(dictionary_type));
        }

        if (!is_row_id_column)
        {
          scan_bind_data->return_types.emplace_back(arrow_type->GetDuckType());
        }

        auto name = string(schema.name);

        // printf("Setting arrow column index %llu to data %s\n", is_row_id_column ? COLUMN_IDENTIFIER_ROW_ID : col_idx, arrow_type->GetDuckType().ToString().c_str());
        scan_bind_data->arrow_table.AddColumn(is_row_id_column ? COLUMN_IDENTIFIER_ROW_ID : col_idx, std::move(arrow_type));

        auto format = string(schema.format);
        if (name.empty())
        {
          name = string("v") + to_string(col_idx);
        }

        if (!is_row_id_column)
        {
          scan_bind_data->names.push_back(name);
        }
      }
    }

    // For each index in the arrow table, the column_ids is asked what
    // where to map that column, the row id can be expressed there.

    // There shouldn't be any projection ids.
    vector<idx_t> projection_ids;

    // Now to initialize the Arrow scan from the reader stream we need to do the steps
    // that the normal table returning function does.

    // bind
    // init global state
    // init local state
    // scan...

    // Init the global state.
    auto scan_global_state = make_uniq<ArrowScanGlobalState>();
    scan_global_state->stream = AirportProduceArrowScan(scan_bind_data->CastNoConst<ArrowScanFunctionData>(), column_ids, nullptr);
    scan_global_state->max_threads = 1;

    // Retain the global state.
    global_state->scan_global_state = std::move(scan_global_state);

    // Now simulate the init input.
    auto fake_init_input = TableFunctionInitInput(
        &scan_bind_data->Cast<FunctionData>(),
        column_ids,
        projection_ids,
        nullptr);

    // Local init.

    auto current_chunk = make_uniq<ArrowArrayWrapper>();
    auto scan_local_state = make_uniq<ArrowScanLocalState>(std::move(current_chunk), context);
    scan_local_state->column_ids = fake_init_input.column_ids;
    scan_local_state->filters = fake_init_input.filters.get();

    global_state->scan_local_state = std::move(scan_local_state);

    // Create a parameter is the commonly passed to the other functions.
    global_state->scan_bind_data = std::move(scan_bind_data);
    global_state->writer = std::move(exchange_result.writer);

    global_state->scan_table_function_input = make_uniq<TableFunctionInput>(
        global_state->scan_bind_data.get(),
        global_state->scan_local_state.get(),
        global_state->scan_global_state.get());
  }
}