#include "airport_extension.hpp"
#include "duckdb.hpp"
#include <arrow/flight/client.h>
#include <arrow/flight/types.h>
#include <arrow/buffer.h>
#include "duckdb/main/extension_util.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "airport_json_common.hpp"
#include "airport_flight_stream.hpp"
#include "airport_macros.hpp"
#include "airport_secrets.hpp"
#include "airport_headers.hpp"
#include "airport_exception.hpp"
#include "duckdb/common/arrow/schema_metadata.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"
#include "airport_take_flight.hpp"

namespace duckdb {

struct TakeFlightParameters {
    string server_location;
    string auth_token;
    string secret_name;
    string database;
};

static TakeFlightParameters ParseTakeFlightParameters(
    const string &server_location,
    ClientContext &context,
    TableFunctionBindInput &input) {
    TakeFlightParameters params;
    params.server_location = server_location;

    for (auto &kv : input.named_parameters) {
        auto loption = StringUtil::Lower(kv.first);
        if (loption == "auth_token") {
            params.auth_token = StringValue::Get(kv.second);
        } else if (loption == "secret") {
            params.secret_name = StringValue::Get(kv.second);
        }
    }

    params.auth_token = AirportAuthTokenForLocation(context, params.server_location, params.secret_name, params.auth_token);
    return params;
}

// [Previous functions take_flight and AirportArrowScanInitGlobal remain unchanged]

static unique_ptr<FunctionData> take_flight_custom_ticket_bind(
    ClientContext &context,
    TableFunctionBindInput &input,
    vector<LogicalType> &return_types,
    vector<string> &names) {
    
    auto server_location = input.inputs[0].ToString();
    auto custom_ticket = input.inputs[1].ToString();
    auto params = ParseTakeFlightParameters(server_location, context, input);

    // Parse JSON ticket to get namespace
    yyjson_doc *doc = yyjson_read(custom_ticket.c_str(), custom_ticket.size(), 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *namespace_val = yyjson_obj_get(root, "namespace_name");
    if (namespace_val) {
        params.database = yyjson_get_str(namespace_val);
    }
    yyjson_doc_free(doc);

    flight::FlightDescriptor descriptor = flight::FlightDescriptor::Command(custom_ticket);

    auto bind_data = make_uniq<AirportTakeFlightBindData>(
        (stream_factory_produce_t)&AirportFlightStreamReader::CreateStream,
        (uintptr_t)nullptr);

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto location, 
        flight::Location::Parse(params.server_location),
        params.server_location,
        descriptor, "");

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto flight_client,
        flight::FlightClient::Connect(location),
        params.server_location,
        descriptor, "");

    arrow::flight::FlightCallOptions call_options;
    airport_add_standard_headers(call_options, params.server_location);
    
    if (!params.auth_token.empty()) {
        call_options.headers.emplace_back("authorization", 
            "Bearer " + params.auth_token);
    }
    
    if (!params.database.empty()) {
        call_options.headers.emplace_back("database", params.database);
    }

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto retrieved_flight_info,
        flight_client->GetFlightInfo(call_options, descriptor),
        params.server_location,
        descriptor, "");

    auto scan_data = make_uniq<AirportTakeFlightScanData>(
        params.server_location,
        std::move(retrieved_flight_info),
        nullptr);

    bind_data->scan_data = std::move(scan_data);
    bind_data->flight_client = std::move(flight_client);
    bind_data->auth_token = params.auth_token;
    bind_data->server_location = params.server_location;
    bind_data->database = params.database;

    auto &data = *bind_data;

    std::shared_ptr<arrow::Schema> info_schema;
    arrow::ipc::DictionaryMemo dictionary_memo;
    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(info_schema,
        bind_data->scan_data->flight_info_->GetSchema(&dictionary_memo),
        params.server_location,
        descriptor, "");

    AIRPORT_ARROW_ASSERT_OK_LOCATION_DESCRIPTOR(
        ExportSchema(*info_schema, &data.schema_root.arrow_schema), 
        params.server_location, 
        descriptor, 
        "ExportSchema");

    for (idx_t col_idx = 0;
         col_idx < (idx_t)data.schema_root.arrow_schema.n_children; col_idx++) {
        auto &schema = *data.schema_root.arrow_schema.children[col_idx];
        if (!schema.release) {
            throw InvalidInputException("airport_take_flight: released schema passed");
        }
        auto arrow_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), schema);

        bool is_row_id_column = false;
        if (schema.metadata != nullptr) {
            auto column_metadata = ArrowSchemaMetadata(schema.metadata);
            auto comment = column_metadata.GetOption("is_row_id");
            if (!comment.empty()) {
                is_row_id_column = true;
                bind_data->row_id_column_index = col_idx;
            }
        }

        if (schema.dictionary) {
            auto dictionary_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), *schema.dictionary);
            if (!is_row_id_column) {
                return_types.emplace_back(dictionary_type->GetDuckType());
            }
            arrow_type->SetDictionary(std::move(dictionary_type));
        } else {
            if (!is_row_id_column) {
                return_types.emplace_back(arrow_type->GetDuckType());
            }
        }

        bind_data->arrow_table.AddColumn(is_row_id_column ? COLUMN_IDENTIFIER_ROW_ID : col_idx, std::move(arrow_type));

        auto name = string(schema.name);
        if (name.empty()) {
            name = string("v") + to_string(col_idx);
        }
        if (!is_row_id_column) {
            names.push_back(name);
        }
    }
    QueryResult::DeduplicateColumns(names);
    return std::move(bind_data);
}

static void take_flight(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
    if (!data_p.local_state) {
        return;
    }
    auto &data = data_p.bind_data->CastNoConst<ArrowScanFunctionData>();
    auto &state = data_p.local_state->Cast<ArrowScanLocalState>();
    auto &global_state = data_p.global_state->Cast<ArrowScanGlobalState>();
    auto &airport_bind_data = data_p.bind_data->Cast<AirportTakeFlightBindData>();

    if (state.chunk_offset >= (idx_t)state.chunk->arrow_array.length) {
        if (!ArrowTableFunction::ArrowScanParallelStateNext(
                context,
                data_p.bind_data.get(),
                state,
                global_state)) {
            return;
        }
    }
    int64_t output_size = MinValue<int64_t>(STANDARD_VECTOR_SIZE,
        state.chunk->arrow_array.length - state.chunk_offset);
    data.lines_read += output_size;

    if (global_state.CanRemoveFilterColumns()) {
        state.all_columns.Reset();
        state.all_columns.SetCardinality(output_size);
        ArrowTableFunction::ArrowToDuckDB(state, data.arrow_table.GetColumns(), 
            state.all_columns, data.lines_read - output_size, false, 
            airport_bind_data.row_id_column_index);
        output.ReferenceColumns(state.all_columns, global_state.projection_ids);
    } else {
        output.SetCardinality(output_size);
        ArrowTableFunction::ArrowToDuckDB(state, data.arrow_table.GetColumns(), output,
            data.lines_read - output_size, false, 
            airport_bind_data.row_id_column_index);
    }

    output.Verify();
    state.chunk_offset += output.size();
}

static unique_ptr<GlobalTableFunctionState> AirportArrowScanInitGlobal(
    ClientContext &context,
    TableFunctionInitInput &input) {
    
    auto &bind_data = input.bind_data->Cast<AirportTakeFlightBindData>();
    auto result = make_uniq<ArrowScanGlobalState>();

    arrow::flight::FlightCallOptions call_options;
    airport_add_standard_headers(call_options, bind_data.server_location);
    
    if (!bind_data.auth_token.empty()) {
        call_options.headers.emplace_back("authorization", 
            "Bearer " + bind_data.auth_token);
    }

    if (!bind_data.database.empty()) {
        call_options.headers.emplace_back("database", bind_data.database);
    }

    auto server_ticket = bind_data.scan_data->flight_info_->endpoints()[0].ticket;

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(
        bind_data.scan_data->stream_,
        bind_data.flight_client->DoGet(
            call_options,
            server_ticket),
        bind_data.server_location,
        bind_data.scan_data->flight_descriptor(),
        "");

    result->stream = AirportProduceArrowScan(bind_data, input.column_ids, 
        input.filters.get());
    result->max_threads = 1;

    return std::move(result);
}

void AddTakeFlightCustomFunction(DatabaseInstance &instance) {
    auto take_flight_custom_function_set = TableFunctionSet("airport_take_flight_custom");

    auto take_flight_custom_function = TableFunction(
        "airport_take_flight_custom",
        {LogicalType::VARCHAR, LogicalType::VARCHAR},
        take_flight,
        take_flight_custom_ticket_bind,
        AirportArrowScanInitGlobal,
        ArrowTableFunction::ArrowScanInitLocal);

    take_flight_custom_function.named_parameters["auth_token"] = LogicalType::VARCHAR;
    take_flight_custom_function.named_parameters["secret"] = LogicalType::VARCHAR;

    take_flight_custom_function_set.AddFunction(take_flight_custom_function);
    ExtensionUtil::RegisterFunction(instance, take_flight_custom_function_set);
}

} // namespace duckdb
