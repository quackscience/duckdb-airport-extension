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
#include "airport_take_flight.hpp"
#include "airport_json_common.hpp"
#include "airport_json_serializer.hpp"
#include "airport_flight_stream.hpp"
#include "airport_macros.hpp"
#include "airport_secrets.hpp"
#include "airport_headers.hpp"
#include "airport_exception.hpp"
#include "duckdb/common/arrow/schema_metadata.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"
#include "msgpack.hpp"

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

  template <typename T>
  static std::vector<std::string> convert_to_strings(const std::vector<T> &vec)
  {
    std::vector<std::string> result(vec.size());
    std::transform(vec.begin(), vec.end(), result.begin(), [](const T &elem)
                   { return std::to_string(elem); });
    return result;
  }

  // Create a FlightDescriptor from a DuckDB value which can be one of a few different
  // types.
  static flight::FlightDescriptor flight_descriptor_from_value(duckdb::Value &flight_descriptor)
  {
    switch (flight_descriptor.type().id())
    {
    case LogicalTypeId::BLOB:
    case LogicalTypeId::VARCHAR:
      return flight::FlightDescriptor::Command(flight_descriptor.ToString());
    case LogicalTypeId::LIST:
    {
      auto &list_values = ListValue::GetChildren(flight_descriptor);
      vector<string> components;
      for (idx_t i = 0; i < list_values.size(); i++)
      {
        auto &child = list_values[i];
        if (child.type().id() != LogicalTypeId::VARCHAR)
        {
          throw InvalidInputException("airport_take_flight: when specifying a path all list components must be a varchar");
        }
        components.emplace_back(child.ToString());
      }
      return flight::FlightDescriptor::Path(components);
    }
    case LogicalTypeId::ARRAY:
    {
      auto &array_values = ArrayValue::GetChildren(flight_descriptor);
      vector<string> components;
      for (idx_t i = 0; i < array_values.size(); i++)
      {
        auto &child = array_values[i];
        if (child.type().id() != LogicalTypeId::VARCHAR)
        {
          throw InvalidInputException("airport_take_flight: when specifying a path all list components must be a varchar");
        }
        components.emplace_back(child.ToString());
      }
      return flight::FlightDescriptor::Path(components);
    }
    // FIXME: deal with the union type returned by Arrow list flights.
    default:
      throw InvalidInputException("airport_take_flight: unknown descriptor type passed");
    }
  }

  AirportTakeFlightParameters AirportParseTakeFlightParameters(
      string &server_location,
      ClientContext &context,
      TableFunctionBindInput &input)
  {
    AirportTakeFlightParameters params;

    params.server_location = server_location;

    for (auto &kv : input.named_parameters)
    {
      auto loption = StringUtil::Lower(kv.first);
      if (loption == "auth_token")
      {
        params.auth_token = StringValue::Get(kv.second);
      }
      else if (loption == "secret")
      {
        params.secret_name = StringValue::Get(kv.second);
      }
      else if (loption == "ticket")
      {
        params.ticket = StringValue::Get(kv.second);
      }
    }

    params.auth_token = AirportAuthTokenForLocation(context, params.server_location, params.secret_name, params.auth_token);
    return params;
  }

  unique_ptr<FunctionData>
  AirportTakeFlightBindWithFlightDescriptor(
      const AirportTakeFlightParameters &take_flight_params,
      const flight::FlightDescriptor &descriptor,
      ClientContext &context,
      TableFunctionBindInput &input,
      vector<LogicalType> &return_types,
      vector<string> &names,
      std::shared_ptr<flight::FlightInfo> *cached_info_ptr,
      std::shared_ptr<GetFlightInfoTableFunctionParameters> table_function_parameters)
  {

    // Create a UID for tracing.
    auto trace_uuid = UUID::ToString(UUID::GenerateRandomUUID());

    D_ASSERT(!take_flight_params.server_location.empty());

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto location, flight::Location::Parse(take_flight_params.server_location),
                                                       take_flight_params.server_location,
                                                       descriptor,
                                                       "");

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(auto flight_client,
                                                       flight::FlightClient::Connect(location),
                                                       take_flight_params.server_location,
                                                       descriptor,
                                                       "");

    arrow::flight::FlightCallOptions call_options;
    airport_add_standard_headers(call_options, take_flight_params.server_location);
    if (!take_flight_params.auth_token.empty())
    {
      std::stringstream ss;
      ss << "Bearer " << take_flight_params.auth_token;
      call_options.headers.emplace_back("authorization", ss.str());
    }

    call_options.headers.emplace_back("airport-trace-id", trace_uuid);

    if (descriptor.type == arrow::flight::FlightDescriptor::PATH)
    {
      auto path_parts = descriptor.path;
      std::string joined_path_parts = join_vector_of_strings(path_parts, '/');
      call_options.headers.emplace_back("airport-flight-path", joined_path_parts);
    }

    // Get the information about the flight, this will allow the
    // endpoint information to be returned.
    unique_ptr<AirportTakeFlightScanData> scan_data;
    if (cached_info_ptr != nullptr)
    {
      scan_data = make_uniq<AirportTakeFlightScanData>(
          take_flight_params.server_location,
          *cached_info_ptr,
          nullptr);
    }
    else
    {
      std::unique_ptr<arrow::flight::FlightInfo> retrieved_flight_info;

      if (table_function_parameters != nullptr)
      {
        // Rather than calling GetFlightInfo we will call DoAction and get
        // get the flight info that way, since it allows us to serialize
        // all of the data we need to send instead of just the flight name.
        std::stringstream packed_buffer;
        msgpack::pack(packed_buffer, table_function_parameters);
        arrow::flight::Action action{"get_flight_info_table_function",
                                     arrow::Buffer::FromString(packed_buffer.str())};

        AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(auto action_results, flight_client->DoAction(call_options, action), take_flight_params.server_location, "airport_get_flight_info_table_function");

        // The only item returned is a serialized flight info.
        AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(auto serialized_flight_info_buffer, action_results->Next(), take_flight_params.server_location, "reading get_flight_info for table function");

        std::string_view serialized_flight_info(reinterpret_cast<const char *>(serialized_flight_info_buffer->body->data()), serialized_flight_info_buffer->body->size());

        // Now deserialize that flight info so we can use it.
        AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(retrieved_flight_info, arrow::flight::FlightInfo::Deserialize(serialized_flight_info), take_flight_params.server_location, "deserialize flight info");

        AIRPORT_ARROW_ASSERT_OK_LOCATION(action_results->Drain(), take_flight_params.server_location, "");
      }
      else
      {
        AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(retrieved_flight_info,
                                                           flight_client->GetFlightInfo(call_options, descriptor),
                                                           take_flight_params.server_location,
                                                           descriptor,
                                                           "");
      }

      // Assert that the descriptor is the same as the one that was passed in.
      if (descriptor != retrieved_flight_info->descriptor())
      {
        throw InvalidInputException("airport_take_flight: descriptor returned from server does not match the descriptor that was passed in to GetFlightInfo, check with Flight server implementation.");
      }

      scan_data = make_uniq<AirportTakeFlightScanData>(
          take_flight_params.server_location,
          std::move(retrieved_flight_info),
          nullptr);
    }

    // Store the flight info on the bind data.

    // After doing a little bit of examination of the DuckDb sources, I learned that
    // that DuckDb supports the "C" interface of Arrow, this means that DuckDB doens't
    // actually have a dependency on Arrow.
    //
    // Arrow Flight requires a dependency on the full Arrow library, because of all of
    // the dependencies.
    //
    // Thankfully there is a "bridge" interface between the C++ based Arrow types returned
    // by the C++ Arrow library and the C based Arrow types that DuckDB already knows how to
    // consume.
    auto ret = make_uniq<AirportTakeFlightBindData>(
        (stream_factory_produce_t)&AirportFlightStreamReader::CreateStream,
        (uintptr_t)scan_data.get());

    ret->scan_data = std::move(scan_data);
    ret->flight_client = std::move(flight_client);
    ret->ticket = take_flight_params.ticket;
    ret->auth_token = take_flight_params.auth_token;
    ret->server_location = take_flight_params.server_location;
    ret->trace_id = trace_uuid;

    auto &data = *ret;

    // Convert the C++ schema into the C format schema, but store it on the bind
    // information
    std::shared_ptr<arrow::Schema> info_schema;
    arrow::ipc::DictionaryMemo dictionary_memo;
    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(info_schema,
                                                       ret->scan_data->flight_info_->GetSchema(&dictionary_memo),
                                                       take_flight_params.server_location,
                                                       descriptor,
                                                       "");

    AIRPORT_ARROW_ASSERT_OK_LOCATION_DESCRIPTOR(ExportSchema(*info_schema, &data.schema_root.arrow_schema), take_flight_params.server_location, descriptor, "ExportSchema");

    for (idx_t col_idx = 0;
         col_idx < (idx_t)data.schema_root.arrow_schema.n_children; col_idx++)
    {
      auto &schema = *data.schema_root.arrow_schema.children[col_idx];
      if (!schema.release)
      {
        throw InvalidInputException("airport_take_flight: released schema passed");
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
          ret->row_id_column_index = col_idx;
        }
      }

      if (schema.dictionary)
      {
        auto dictionary_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), *schema.dictionary);
        arrow_type->SetDictionary(std::move(dictionary_type));
      }

      if (!is_row_id_column)
      {
        return_types.emplace_back(arrow_type->GetDuckType());
      }

      ret->arrow_table.AddColumn(is_row_id_column ? COLUMN_IDENTIFIER_ROW_ID : col_idx, std::move(arrow_type));

      // FIXME: determine if bind should include the row_id column.

      auto format = string(schema.format);
      auto name = string(schema.name);
      if (name.empty())
      {
        name = string("v") + to_string(col_idx);
      }
      if (!is_row_id_column)
      {
        names.push_back(name);
      }
      // printf("Added column with id %lld %s\n", is_row_id_column ? COLUMN_IDENTIFIER_ROW_ID : col_idx, name.c_str());
    }
    QueryResult::DeduplicateColumns(names);
    return std::move(ret);
  }

  static unique_ptr<FunctionData> take_flight_bind(
      ClientContext &context,
      TableFunctionBindInput &input,
      vector<LogicalType> &return_types,
      vector<string> &names)
  {
    auto server_location = input.inputs[0].ToString();
    auto params = AirportParseTakeFlightParameters(server_location, context, input);
    auto descriptor = flight_descriptor_from_value(input.inputs[1]);

    return AirportTakeFlightBindWithFlightDescriptor(
        params,
        descriptor,
        context,
        input, return_types, names, nullptr, nullptr);
  }

  static unique_ptr<FunctionData> take_flight_bind_with_pointer(
      ClientContext &context,
      TableFunctionBindInput &input,
      vector<LogicalType> &return_types,
      vector<string> &names)
  {
    auto server_location = input.inputs[0].ToString();
    auto params = AirportParseTakeFlightParameters(server_location, context, input);

    if (input.inputs[1].IsNull())
    {
      throw BinderException("airport: take_flight_with_pointer, pointers to flight descriptor cannot be null");
    }

    const auto info = reinterpret_cast<std::shared_ptr<flight::FlightInfo> *>(input.inputs[1].GetPointer());

    return AirportTakeFlightBindWithFlightDescriptor(
        params,
        info->get()->descriptor(),
        context,
        input,
        return_types,
        names,
        info,
        nullptr);
  }

  void AirportTakeFlight(ClientContext &context, TableFunctionInput &data_p, DataChunk &output)
  {
    if (!data_p.local_state)
    {
      return;
    }
    auto &data = data_p.bind_data->CastNoConst<ArrowScanFunctionData>();
    auto &state = data_p.local_state->Cast<ArrowScanLocalState>();
    auto &global_state = data_p.global_state->Cast<ArrowScanGlobalState>();
    auto &airport_bind_data = data_p.bind_data->Cast<AirportTakeFlightBindData>();

    //! Out of tuples in this chunk
    if (state.chunk_offset >= (idx_t)state.chunk->arrow_array.length)
    {
      if (!ArrowTableFunction::ArrowScanParallelStateNext(
              context,
              data_p.bind_data.get(),
              state,
              global_state))
      {
        return;
      }
    }
    int64_t output_size =
        MinValue<int64_t>(STANDARD_VECTOR_SIZE,
                          state.chunk->arrow_array.length - state.chunk_offset);
    data.lines_read += output_size;

    if (global_state.CanRemoveFilterColumns())
    {
      state.all_columns.Reset();
      state.all_columns.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(state, data.arrow_table.GetColumns(), state.all_columns,
                                        data.lines_read - output_size, false, airport_bind_data.row_id_column_index);
      output.ReferenceColumns(state.all_columns, global_state.projection_ids);
    }
    else
    {
      output.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(state, data.arrow_table.GetColumns(), output,
                                        data.lines_read - output_size, false, airport_bind_data.row_id_column_index);
    }

    output.Verify();
    state.chunk_offset += output.size();
  }

  static unique_ptr<NodeStatistics> take_flight_cardinality(ClientContext &context, const FunctionData *data)
  {
    // To estimate the cardinality of the flight, we can peek at the flight information
    // that was retrieved during the bind function.
    //
    // This estimate does not take into account any filters that may have been applied
    //
    auto &bind_data = data->Cast<AirportTakeFlightBindData>();
    auto flight_estimated_records = bind_data.scan_data.get()->total_records();

    if (flight_estimated_records != -1)
    {
      return make_uniq<NodeStatistics>(flight_estimated_records);
    }
    return make_uniq<NodeStatistics>(100000);
  }

  static void take_flight_complex_filter_pushdown(ClientContext &context, LogicalGet &get, FunctionData *bind_data_p,
                                                  vector<unique_ptr<Expression>> &filters)
  {
    auto allocator = AirportJSONAllocator(BufferAllocator::Get(context));

    auto alc = allocator.GetYYAlc();

    auto doc = AirportJSONCommon::CreateDocument(alc);
    auto result_obj = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, result_obj);

    auto filters_arr = yyjson_mut_arr(doc);

    for (auto &f : filters)
    {
      auto serializer = AirportJsonSerializer(doc, false, false, false);
      f->Serialize(serializer);
      yyjson_mut_arr_append(filters_arr, serializer.GetRootObject());
    }

    yyjson_mut_val *column_id_names = yyjson_mut_arr(doc);
    for (auto id : get.GetColumnIds())
    {
      // So there can be a special column id specified called rowid.
      yyjson_mut_arr_add_str(doc, column_id_names, id.IsRowIdColumn() ? "rowid" : get.names[id.GetPrimaryIndex()].c_str());
    }

    yyjson_mut_obj_add_val(doc, result_obj, "filters", filters_arr);
    yyjson_mut_obj_add_val(doc, result_obj, "column_binding_names_by_index", column_id_names);
    idx_t len;
    auto data = yyjson_mut_val_write_opts(
        result_obj,
        AirportJSONCommon::WRITE_FLAG,
        alc, reinterpret_cast<size_t *>(&len), nullptr);

    if (data == nullptr)
    {
      throw SerializationException(
          "Failed to serialize json, perhaps the query contains invalid utf8 characters?");
    }

    auto json_result = string(data, (size_t)len);

    auto &bind_data = bind_data_p->Cast<AirportTakeFlightBindData>();

    bind_data.json_filters = json_result;
  }

  unique_ptr<ArrowArrayStreamWrapper> AirportProduceArrowScan(const ArrowScanFunctionData &function,
                                                              const vector<column_t> &column_ids, TableFilterSet *filters)
  {
    //! Generate Projection Pushdown Vector
    ArrowStreamParameters parameters;
    for (idx_t idx = 0; idx < column_ids.size(); idx++)
    {
      auto col_idx = column_ids[idx];
      if (col_idx != COLUMN_IDENTIFIER_ROW_ID)
      {
        auto &schema = *function.schema_root.arrow_schema.children[col_idx];
        parameters.projected_columns.projection_map[idx] = schema.name;
        parameters.projected_columns.columns.emplace_back(schema.name);
        parameters.projected_columns.filter_to_col[idx] = col_idx;
      }
    }
    parameters.filters = filters;

    return function.scanner_producer(function.stream_factory_ptr, parameters);
  }

  static string CompressString(const string &input, const string &location, const flight::FlightDescriptor &descriptor)
  {
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

  static string BuildDynamicTicketData(const string &json_filters, const string &column_ids, uint32_t *uncompressed_length, const string &location, const flight::FlightDescriptor &descriptor)
  {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    // Add key-value pairs to the JSON object
    if (!json_filters.empty())
    {
      yyjson_mut_obj_add_str(doc, root, "airport-duckdb-json-filters", json_filters.c_str());
    }
    if (!column_ids.empty())
    {
      yyjson_mut_obj_add_str(doc, root, "airport-duckdb-column-ids", column_ids.c_str());
    }

    // Serialize the JSON document to a string
    char *metadata_str = yyjson_mut_write(doc, 0, nullptr);

    auto metadata_doc_string = string(metadata_str);

    *uncompressed_length = metadata_doc_string.size();

    auto compressed_metadata = CompressString(metadata_doc_string, location, descriptor);
    free(metadata_str);
    yyjson_mut_doc_free(doc);

    return compressed_metadata;
  }

  unique_ptr<GlobalTableFunctionState> AirportArrowScanInitGlobal(ClientContext &context,
                                                                  TableFunctionInitInput &input)
  {
    auto &bind_data = input.bind_data->Cast<AirportTakeFlightBindData>();

    auto result = make_uniq<ArrowScanGlobalState>();

    arrow::flight::FlightCallOptions call_options;
    airport_add_standard_headers(call_options, bind_data.server_location);

    // Since the ticket is an opaque reference to the stream, its useful to the middle ware
    // sometimes to know what the path of the flight is.
    auto descriptor = bind_data.scan_data->flight_descriptor();
    if (descriptor.type == arrow::flight::FlightDescriptor::PATH)
    {
      auto path_parts = descriptor.path;
      std::string joined_path_parts = join_vector_of_strings(path_parts, '/');
      call_options.headers.emplace_back("airport-flight-path", joined_path_parts);
    }

    if (!bind_data.auth_token.empty())
    {
      std::stringstream ss;
      ss << "Bearer " << bind_data.auth_token;
      call_options.headers.emplace_back("authorization", ss.str());
    }

    call_options.headers.emplace_back("airport-trace-id", bind_data.trace_id);

    if (bind_data.skip_producing_result_for_update_or_delete)
    {
      // This is a special case where the result of the scan should be skipped.
      // This is useful when the scan is being used to update or delete rows.
      // For a table that doesn't actually produce row ids, so filtering cannot be applied.
      call_options.headers.emplace_back("airport-skip-producing-results", "1");
    }

    // Rather than using the headers, check to see if the ticket starts with <TICKET_ALLOWS_METADATA>

    auto server_ticket = bind_data.scan_data->flight_info_->endpoints()[0].ticket;
    auto server_ticket_contents = server_ticket.ticket;

    if (!bind_data.ticket.empty())
    {
      server_ticket = flight::Ticket{bind_data.ticket};
    }
    else if (server_ticket_contents.find("<TICKET_ALLOWS_METADATA>") == 0)
    {
      // This ticket allows metadata to be supplied in the ticket.

      auto ticket_without_preamble = server_ticket_contents.substr(strlen("<TICKET_ALLOWS_METADATA>"));

      // encode the length as a unsigned int32 in network byte order
      uint32_t ticket_length = ticket_without_preamble.size();
      auto ticket_length_bytes = std::string((char *)&ticket_length, sizeof(ticket_length));

      uint32_t uncompressed_length;
      // So the column ids can be sent here but there is a special one.
      // COLUMN_IDENTIFIER_ROW_ID that will be sent.
      auto joined_column_ids = join_vector_of_strings(convert_to_strings(input.column_ids), ',');

      auto dynamic_ticket = BuildDynamicTicketData(bind_data.json_filters, joined_column_ids, &uncompressed_length, bind_data.server_location,
                                                   bind_data.scan_data->flight_descriptor());

      auto compressed_length_bytes = std::string((char *)&uncompressed_length, sizeof(uncompressed_length));

      auto manipulated_ticket_data = "<TICKET_WITH_METADATA>" + ticket_length_bytes + ticket_without_preamble + compressed_length_bytes + dynamic_ticket;

      server_ticket = flight::Ticket{manipulated_ticket_data};
    }

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(
        bind_data.scan_data->stream_,
        bind_data.flight_client->DoGet(
            call_options,
            server_ticket),
        bind_data.server_location,
        bind_data.scan_data->flight_descriptor(),
        "");

    //    bind_data.scan_data->stream_ = std::move(flight_stream);
    result->stream = AirportProduceArrowScan(bind_data, input.column_ids, input.filters.get());

    // Since we're single threaded, we can only really use a single thread at a time.
    result->max_threads = 1;
    // if (input.CanRemoveFilterColumns())
    // {
    //   result->projection_ids = input.projection_ids;
    //   for (const auto &col_idx : input.column_ids)
    //   {
    //     if (col_idx == COLUMN_IDENTIFIER_ROW_ID)
    //     {
    //       result->scanned_types.emplace_back(LogicalTypeId::BIGINT);
    //     }
    //     else
    //     {
    //       result->scanned_types.push_back(bind_data.all_types[col_idx]);
    //     }
    //   }
    // }

    // So right now the scanned_types is null

    return std::move(result);
  }

  static double take_flight_scan_progress(ClientContext &, const FunctionData *data, const GlobalTableFunctionState *global_state)
  {
    auto &bind_data = data->Cast<AirportTakeFlightBindData>();
    lock_guard<mutex> guard(bind_data.lock);

    return bind_data.scan_data->progress_ * 100.0;
  }

  void AddTakeFlightFunction(DatabaseInstance &instance)
  {

    auto take_flight_function_set = TableFunctionSet("airport_take_flight");

    auto take_flight_function_with_descriptor = TableFunction(
        "airport_take_flight",
        {LogicalType::VARCHAR, LogicalType::ANY},
        AirportTakeFlight,
        take_flight_bind,
        AirportArrowScanInitGlobal,
        ArrowTableFunction::ArrowScanInitLocal);

    take_flight_function_with_descriptor.named_parameters["auth_token"] = LogicalType::VARCHAR;
    take_flight_function_with_descriptor.named_parameters["secret"] = LogicalType::VARCHAR;
    take_flight_function_with_descriptor.named_parameters["ticket"] = LogicalType::BLOB;
    take_flight_function_with_descriptor.pushdown_complex_filter = take_flight_complex_filter_pushdown;

    // Add support for optional named paraemters that would be appended to the descriptor
    // of the flight, ideally parameters would be JSON encoded.

    take_flight_function_with_descriptor.cardinality = take_flight_cardinality;
    //    take_flight_function_with_descriptor.get_batch_index = nullptr;
    take_flight_function_with_descriptor.projection_pushdown = true;
    take_flight_function_with_descriptor.filter_pushdown = false;
    take_flight_function_with_descriptor.table_scan_progress = take_flight_scan_progress;
    take_flight_function_set.AddFunction(take_flight_function_with_descriptor);

    auto take_flight_function_with_pointer = TableFunction(
        "airport_take_flight",
        {LogicalType::VARCHAR, LogicalType::POINTER},
        AirportTakeFlight,
        take_flight_bind_with_pointer,
        AirportArrowScanInitGlobal,
        ArrowTableFunction::ArrowScanInitLocal);

    take_flight_function_with_pointer.named_parameters["auth_token"] = LogicalType::VARCHAR;
    take_flight_function_with_pointer.named_parameters["secret"] = LogicalType::VARCHAR;
    take_flight_function_with_pointer.pushdown_complex_filter = take_flight_complex_filter_pushdown;

    // Add support for optional named paraemters that would be appended to the descriptor
    // of the flight, ideally parameters would be JSON encoded.

    take_flight_function_with_pointer.cardinality = take_flight_cardinality;
    //    take_flight_function_with_pointer.get_batch_index = nullptr;
    take_flight_function_with_pointer.projection_pushdown = true;
    take_flight_function_with_pointer.filter_pushdown = false;
    take_flight_function_with_pointer.table_scan_progress = take_flight_scan_progress;

    take_flight_function_set.AddFunction(take_flight_function_with_pointer);

    ExtensionUtil::RegisterFunction(instance, take_flight_function_set);
  }
}