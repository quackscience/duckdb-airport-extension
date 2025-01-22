#include "airport_extension.hpp"
#include "duckdb.hpp"

// Arrow includes.
#include <arrow/flight/client.h>
#include <arrow/flight/types.h>
#include <arrow/buffer.h>

#include "duckdb/main/extension_util.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/common/types/uuid.hpp"

#include "airport_json_common.hpp"
#include "airport_json_serializer.hpp"
#include "airport_flight_stream.hpp"
#include "airport_macros.hpp"
#include "airport_secrets.hpp"
#include "airport_headers.hpp"
#include "airport_exception.hpp"
#include "duckdb/common/arrow/schema_metadata.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"
#include "airport_take_flight.hpp"

namespace duckdb {

static std::string join_vector_of_strings(const std::vector<std::string> &vec, const char joiner) {
    if (vec.empty())
        return "";

    return std::accumulate(
        std::next(vec.begin()), vec.end(), vec.front(),
        [joiner](const std::string &a, const std::string &b) {
            return a + joiner + b;
        });
}

template <typename T>
static std::vector<std::string> convert_to_strings(const std::vector<T> &vec) {
    std::vector<std::string> result(vec.size());
    std::transform(vec.begin(), vec.end(), result.begin(), [](const T &elem) { return std::to_string(elem); });
    return result;
}

static flight::FlightDescriptor flight_custom_ticket_descriptor(const std::string &custom_ticket) {
    return flight::FlightDescriptor::Command(custom_ticket);
}

static string CompressString(const string &input, const string &location, const flight::FlightDescriptor &descriptor) {
    auto codec = arrow::util::Codec::Create(arrow::Compression::ZSTD, 1).ValueOrDie();

    // Estimate the maximum compressed size (usually larger than original size)
    int64_t max_compressed_len = codec->MaxCompressedLen(input.size(), reinterpret_cast<const uint8_t *>(input.data()));

    // Allocate a buffer to hold the compressed data

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto compressed_buffer, arrow::AllocateBuffer(max_compressed_len), location, descriptor, "");

    // Perform the compression
    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto compressed_size,
                                                       codec->Compress(
                                                           input.size(),
                                                           reinterpret_cast<const uint8_t *>(input.data()),
                                                           max_compressed_len,
                                                           compressed_buffer->mutable_data()),
                                                       location, descriptor, "");

    // If you want to write the compressed data to a string
    std::string compressed_str(reinterpret_cast<const char *>(compressed_buffer->data()), compressed_size);
    return compressed_str;
}


static string BuildDynamicTicketData(const string &json_filters, const string &column_ids, uint32_t *uncompressed_length, const string &location, const flight::FlightDescriptor &descriptor) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    if (!json_filters.empty()) {
        yyjson_mut_obj_add_str(doc, root, "airport-duckdb-json-filters", json_filters.c_str());
    }
    if (!column_ids.empty()) {
        yyjson_mut_obj_add_str(doc, root, "airport-duckdb-column-ids", column_ids.c_str());
    }

    char *metadata_str = yyjson_mut_write(doc, 0, nullptr);
    auto metadata_doc_string = string(metadata_str);
    *uncompressed_length = metadata_doc_string.size();
    auto compressed_metadata = CompressString(metadata_doc_string, location, descriptor);
    free(metadata_str);
    yyjson_mut_doc_free(doc);
    return compressed_metadata;
}

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
        } else if (loption == "database") {
            params.database = StringValue::Get(kv.second);
        }
    }

    params.auth_token = AirportAuthTokenForLocation(context, params.server_location, params.secret_name, params.auth_token);
    return params;
}

static unique_ptr<FunctionData> take_flight_bind_with_descriptor(
    TakeFlightParameters &take_flight_params,
    const flight::FlightDescriptor &descriptor,
    ClientContext &context,
    TableFunctionBindInput &input,
    vector<LogicalType> &return_types,
    vector<string> &names,
    std::shared_ptr<flight::FlightInfo> *cached_info_ptr) {

    auto trace_uuid = UUID::ToString(UUID::GenerateRandomUUID());
    D_ASSERT(!take_flight_params.server_location.empty());

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto location, flight::Location::Parse(take_flight_params.server_location),
                                                       take_flight_params.server_location,
                                                       descriptor, "");

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto flight_client,
                                                       flight::FlightClient::Connect(location),
                                                       take_flight_params.server_location,
                                                       descriptor, "");

    arrow::flight::FlightCallOptions call_options;
    airport_add_standard_headers(call_options, take_flight_params.server_location);
    if (!take_flight_params.auth_token.empty()) {
        std::stringstream ss;
        ss << "Bearer " << take_flight_params.auth_token;
        call_options.headers.emplace_back("authorization", ss.str());
    }

    call_options.headers.emplace_back("airport-trace-id", trace_uuid);

    if (descriptor.type == arrow::flight::FlightDescriptor::PATH) {
        auto path_parts = descriptor.path;
        std::string joined_path_parts = join_vector_of_strings(path_parts, '/');
        call_options.headers.emplace_back("airport-flight-path", joined_path_parts);
    }

    unique_ptr<AirportTakeFlightScanData> scan_data;
    if (cached_info_ptr != nullptr) {
        scan_data = make_uniq<AirportTakeFlightScanData>(
            take_flight_params.server_location,
            *cached_info_ptr,
            nullptr);
    } else {
        AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto retrieved_flight_info,
                                                           flight_client->GetFlightInfo(call_options, descriptor),
                                                           take_flight_params.server_location,
                                                           descriptor, "");

        if (descriptor != retrieved_flight_info->descriptor()) {
            throw InvalidInputException("airport_take_flight: descriptor returned from server does not match the descriptor that was passed in to GetFlightInfo, check with Flight server implementation");
        }

        scan_data = make_uniq<AirportTakeFlightScanData>(
            take_flight_params.server_location,
            std::move(retrieved_flight_info),
            nullptr);
    }

    auto ret = make_uniq<AirportTakeFlightBindData>(
        (stream_factory_produce_t)&AirportFlightStreamReader::CreateStream,
        (uintptr_t)scan_data.get());

    ret->scan_data = std::move(scan_data);
    ret->flight_client = std::move(flight_client);
    ret->auth_token = take_flight_params.auth_token;
    ret->server_location = take_flight_params.server_location;
    ret->database = take_flight_params.database;
    ret->trace_id = trace_uuid;

    auto &data = *ret;

    std::shared_ptr<arrow::Schema> info_schema;
    arrow::ipc::DictionaryMemo dictionary_memo;
    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(info_schema,
                                                       ret->scan_data->flight_info_->GetSchema(&dictionary_memo),
                                                       take_flight_params.server_location,
                                                       descriptor, "");

    AIRPORT_ARROW_ASSERT_OK_LOCATION_DESCRIPTOR(ExportSchema(*info_schema, &data.schema_root.arrow_schema), take_flight_params.server_location, descriptor, "ExportSchema");

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
                ret->row_id_column_index = col_idx;
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

        ret->arrow_table.AddColumn(is_row_id_column ? COLUMN_IDENTIFIER_ROW_ID : col_idx, std::move(arrow_type));

        auto format = string(schema.format);
        auto name = string(schema.name);
        if (name.empty()) {
            name = string("v") + to_string(col_idx);
        }
        if (!is_row_id_column) {
            names.push_back(name);
        }
    }
    QueryResult::DeduplicateColumns(names);
    return std::move(ret);
}

static unique_ptr<FunctionData> take_flight_custom_ticket_bind(
    ClientContext &context,
    TableFunctionBindInput &input,
    vector<LogicalType> &return_types,
    vector<string> &names) {
    auto server_location = input.inputs[0].ToString();
    auto custom_ticket = input.inputs[1].ToString();
    auto params = ParseTakeFlightParameters(server_location, context, input);

    // Extract database value from the custom ticket JSON
    yyjson_doc *doc = yyjson_read(custom_ticket.c_str(), custom_ticket.size(), 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *database_val = yyjson_obj_get(root, "database");
    string database = yyjson_get_str(database_val);
    yyjson_doc_free(doc);

    flight::FlightDescriptor descriptor = flight_custom_ticket_descriptor(custom_ticket);

    auto bind_data = take_flight_bind_with_descriptor(params, descriptor, context, input, return_types, names, nullptr);

    // Add the database to the bind data
    auto &airport_bind_data = bind_data->CastNoConst<AirportTakeFlightBindData>();
    airport_bind_data.database = database;

    return bind_data;
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
    int64_t output_size =
        MinValue<int64_t>(STANDARD_VECTOR_SIZE,
                          state.chunk->arrow_array.length - state.chunk_offset);
    data.lines_read += output_size;

    if (global_state.CanRemoveFilterColumns()) {
        state.all_columns.Reset();
        state.all_columns.SetCardinality(output_size);
        ArrowTableFunction::ArrowToDuckDB(state, data.arrow_table.GetColumns(), state.all_columns,
                                          data.lines_read - output_size, false, airport_bind_data.row_id_column_index);
        output.ReferenceColumns(state.all_columns, global_state.projection_ids);
    } else {
        output.SetCardinality(output_size);
        ArrowTableFunction::ArrowToDuckDB(state, data.arrow_table.GetColumns(), output,
                                          data.lines_read - output_size, false, airport_bind_data.row_id_column_index);
    }

    output.Verify();
    state.chunk_offset += output.size();
}

static unique_ptr<GlobalTableFunctionState> AirportArrowScanInitGlobal(ClientContext &context,
                                                                       TableFunctionInitInput &input) {
    auto &bind_data = input.bind_data->Cast<AirportTakeFlightBindData>();

    auto result = make_uniq<ArrowScanGlobalState>();

    arrow::flight::FlightCallOptions call_options;
    airport_add_standard_headers(call_options, bind_data.server_location);

    auto descriptor = bind_data.scan_data->flight_descriptor();
    if (descriptor.type == arrow::flight::FlightDescriptor::PATH) {
        auto path_parts = descriptor.path;
        std::string joined_path_parts = join_vector_of_strings(path_parts, '/');
        call_options.headers.emplace_back("airport-flight-path", joined_path_parts);
    }

    if (!bind_data.auth_token.empty()) {
        std::stringstream ss;
        ss << "Bearer " << bind_data.auth_token;
        call_options.headers.emplace_back("authorization", ss.str());
    }

    if (!bind_data.database.empty()) {
        call_options.headers.emplace_back("database", bind_data.database);
    }

    call_options.headers.emplace_back("airport-trace-id", bind_data.trace_id);

    if (bind_data.skip_producing_result_for_update_or_delete) {
        call_options.headers.emplace_back("airport-skip-producing-results", "1");
    }

    auto server_ticket = bind_data.scan_data->flight_info_->endpoints()[0].ticket;
    auto server_ticket_contents = server_ticket.ticket;
    if (server_ticket_contents.find("<TICKET_ALLOWS_METADATA>") == 0) {
        auto ticket_without_preamble = server_ticket_contents.substr(strlen("<TICKET_ALLOWS_METADATA>"));

        uint32_t ticket_length = ticket_without_preamble.size();
        auto ticket_length_bytes = std::string((char *)&ticket_length, sizeof(ticket_length));

        uint32_t uncompressed_length;
        auto joined_column_ids = join_vector_of_strings(convert_to_strings(input.column_ids), ',');

        auto dynamic_ticket = BuildDynamicTicketData(bind_data.json_filters, joined_column_ids, &uncompressed_length, bind_data.server_location,
                                                     bind_data.scan_data->flight_descriptor());

        auto compressed_length_bytes = std::string((char *)&uncompressed_length, sizeof(uncompressed_length));

        auto manipulated_ticket_data = "<TICKET_WITH_METADATA>" + ticket_length_bytes + ticket_without_preamble + compressed_length_bytes + dynamic_ticket;

        server_ticket = flight::Ticket(manipulated_ticket_data);
    }

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(
        bind_data.scan_data->stream_,
        bind_data.flight_client->DoGet(
            call_options,
            server_ticket),
        bind_data.server_location,
        bind_data.scan_data->flight_descriptor(),
        "");

    result->stream = AirportProduceArrowScan(bind_data, input.column_ids, input.filters.get());

    result->max_threads = 1;

    return std::move(result);
}

void AddTakeFlightCustomFunction(DatabaseInstance &instance) {

    auto take_flight_custom_function_set = TableFunctionSet("airport_take_flight_custom");

    auto take_flight_custom_function = TableFunction(
        "airport_take_flight_custom",
        {LogicalType::VARCHAR, LogicalType::VARCHAR},  // Update the second parameter to VARCHAR
        take_flight,
        take_flight_custom_ticket_bind,
        AirportArrowScanInitGlobal,
        ArrowTableFunction::ArrowScanInitLocal);

    take_flight_custom_function.named_parameters["auth_token"] = LogicalType::VARCHAR;
    take_flight_custom_function.named_parameters["secret"] = LogicalType::VARCHAR;
    take_flight_custom_function.named_parameters["database"] = LogicalType::VARCHAR;

    take_flight_custom_function_set.AddFunction(take_flight_custom_function);

    ExtensionUtil::RegisterFunction(instance, take_flight_custom_function_set);
}

} // namespace duckdb
