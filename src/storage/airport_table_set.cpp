#include "storage/airport_catalog_api.hpp"
#include "storage/airport_catalog.hpp"
#include "storage/airport_table_set.hpp"
#include "storage/airport_transaction.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/constraints/not_null_constraint.hpp"
#include "duckdb/parser/constraints/unique_constraint.hpp"
#include "duckdb/parser/constraints/check_constraint.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/planner/parsed_data/bound_create_table_info.hpp"
#include "duckdb/catalog/dependency_list.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parsed_data/create_function_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/parser/constraints/list.hpp"
#include "storage/airport_schema_entry.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb.hpp"
#include "airport_macros.hpp"
#include <arrow/c/bridge.h>
#include <arrow/util/key_value_metadata.h>
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/common/arrow/schema_metadata.hpp"
#include "storage/airport_curl_pool.hpp"
#include "airport_macros.hpp"
#include "airport_secrets.hpp"
#include "airport_headers.hpp"
#include "airport_scalar_function.hpp"
#include "yyjson.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "airport_take_flight.hpp"
#include "duckdb/common/arrow/arrow_appender.hpp"
#include <arrow/buffer.h>
#include <arrow/io/memory.h>

using namespace duckdb_yyjson; // NOLINT

struct FunctionCatalogSchemaName
{
  std::string catalog_name;
  std::string schema_name;
  std::string name;

  // Define equality operator to compare two keys
  bool operator==(const FunctionCatalogSchemaName &other) const
  {
    return catalog_name == other.catalog_name && schema_name == other.schema_name && name == other.name;
  }
};

namespace std
{
  template <>
  struct hash<FunctionCatalogSchemaName>
  {
    size_t operator()(const FunctionCatalogSchemaName &k) const
    {
      // Combine the hash of all 3 strings
      return hash<std::string>()(k.catalog_name) ^ (hash<std::string>()(k.schema_name) << 1) ^ (hash<std::string>()(k.name) << 2);
    }
  };
}

namespace duckdb
{

  AirportTableFunctionSet::AirportTableFunctionSet(AirportCurlPool &connection_pool, AirportSchemaEntry &schema, const string &cache_directory_) : AirportInSchemaSet(schema), connection_pool(connection_pool), cache_directory(cache_directory_)
  {
  }

  AirportScalarFunctionSet::AirportScalarFunctionSet(AirportCurlPool &connection_pool, AirportSchemaEntry &schema, const string &cache_directory_) : AirportInSchemaSet(schema), connection_pool(connection_pool), cache_directory(cache_directory_)
  {
  }

  AirportTableSet::AirportTableSet(AirportCurlPool &connection_pool, AirportSchemaEntry &schema, const string &cache_directory_) : AirportInSchemaSet(schema), connection_pool(connection_pool), cache_directory(cache_directory_)
  {
  }

  void AirportTableSet::LoadEntries(ClientContext &context)
  {
    // auto &transaction = AirportTransaction::Get(context, catalog);

    auto &airport_catalog = catalog.Cast<AirportCatalog>();

    // TODO: handle out-of-order columns using position property
    auto curl = connection_pool.acquire();
    auto contents = AirportAPI::GetSchemaItems(
        curl,
        catalog.GetDBPath(),
        schema.name,
        schema.schema_data->contents_url,
        schema.schema_data->contents_sha256,
        schema.schema_data->contents_serialized,
        cache_directory,
        airport_catalog.credentials);
    connection_pool.release(curl);

    //    printf("AirportTableSet loading entries\n");
    //    printf("Total tables: %lu\n", tables_and_functions.first.size());

    for (auto &table : contents->tables)
    {
      // D_ASSERT(schema.name == table.schema_name);
      CreateTableInfo info;

      info.table = table.name;
      info.comment = table.comment;

      std::shared_ptr<arrow::Schema> info_schema;
      arrow::ipc::DictionaryMemo dictionary_memo;
      string error_location = "(" + airport_catalog.credentials.location + ")";
      AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(info_schema, table.flight_info->GetSchema(&dictionary_memo), airport_catalog.credentials.location, table.flight_info->descriptor(), "");

      ArrowSchema arrow_schema;

      AIRPORT_ARROW_ASSERT_OK_LOCATION_DESCRIPTOR(ExportSchema(*info_schema, &arrow_schema), airport_catalog.credentials.location, table.flight_info->descriptor(), "ExportSchema");

      vector<string> column_names;
      vector<duckdb::LogicalType> return_types;
      vector<string> not_null_columns;

      LogicalType row_id_type = LogicalType(LogicalType::ROW_TYPE);

      if (arrow_schema.metadata != nullptr)
      {
        auto schema_metadata = ArrowSchemaMetadata(arrow_schema.metadata);

        auto check_constraints = schema_metadata.GetOption("check_constraints");
        if (!check_constraints.empty())
        {
          yyjson_doc *doc = yyjson_read(check_constraints.c_str(), check_constraints.size(), 0);
          if (doc)
          {
            // Get the root of the JSON document
            yyjson_val *root = yyjson_doc_get_root(doc);

            // Ensure the root is an object (dictionary)
            if (root && yyjson_is_arr(root))
            {
              size_t idx, max;
              yyjson_val *val;
              yyjson_arr_foreach(root, idx, max, val)
              {
                if (yyjson_is_str(val))
                {
                  string expression = yyjson_get_str(val);

                  auto expression_list = Parser::ParseExpressionList(expression, context.GetParserOptions());
                  if (expression_list.size() != 1)
                  {
                    throw ParserException("Failed to parse CHECK constraint expression: " + expression + " for table " + table.name);
                  }
                  info.constraints.emplace_back(make_uniq<CheckConstraint>(std::move(expression_list[0])));
                }
                else
                {
                  yyjson_doc_free(doc);
                  throw ParserException("Encountered non string element in CHECK constraints for table  " + table.name);
                }
              }
            }
            else
            {
              // Free the JSON document after parsing
              yyjson_doc_free(doc);
              throw ParserException("Failed to parse check constraints JSON for table " + table.name + " is not an array");
            }
            // Free the JSON document after parsing
            yyjson_doc_free(doc);
          }
          else
          {
            throw ParserException("Failed to parse check constraints as JSON for table " + table.name);
          }
        }
      }

      for (idx_t col_idx = 0;
           col_idx < (idx_t)arrow_schema.n_children; col_idx++)
      {
        auto &column = *arrow_schema.children[col_idx];
        if (!column.release)
        {
          throw InvalidInputException("AirportTableSet::LoadEntries: released schema passed");
        }

        if (column.metadata != nullptr)
        {
          auto column_metadata = ArrowSchemaMetadata(column.metadata);

          auto is_row_id = column_metadata.GetOption("is_row_id");
          if (!is_row_id.empty())
          {
            row_id_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), column)->GetDuckType();

            // So the skipping here is a problem, since its assumed
            // that the return_type and column_names can be easily indexed.
            continue;
          }
        }

        auto column_name = string(column.name);
        if (column_name.empty())
        {
          column_name = string("v") + to_string(col_idx);
        }

        column_names.emplace_back(column_name);

        auto arrow_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), column);
        if (column.dictionary)
        {
          auto dictionary_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), *column.dictionary);
          return_types.emplace_back(dictionary_type->GetDuckType());
        }
        else
        {
          return_types.emplace_back(arrow_type->GetDuckType());
        }

        if (!(column.flags & ARROW_FLAG_NULLABLE))
        {
          not_null_columns.emplace_back(column_name);
        }
      }

      QueryResult::DeduplicateColumns(column_names);
      idx_t row_id_adjust = 0;
      for (idx_t col_idx = 0;
           col_idx < (idx_t)arrow_schema.n_children; col_idx++)
      {
        auto &column = *arrow_schema.children[col_idx];
        if (column.metadata != nullptr)
        {
          auto column_metadata = ArrowSchemaMetadata(column.metadata);

          auto is_row_id = column_metadata.GetOption("is_row_id");
          if (!is_row_id.empty())
          {
            row_id_adjust = 1;
            continue;
          }
        }

        auto column_def = ColumnDefinition(column_names[col_idx - row_id_adjust], return_types[col_idx - row_id_adjust]);
        if (column.metadata != nullptr)
        {
          auto column_metadata = ArrowSchemaMetadata(column.metadata);

          auto comment = column_metadata.GetOption("comment");
          if (!comment.empty())
          {
            column_def.SetComment(duckdb::Value(comment));
          }

          auto default_value = column_metadata.GetOption("default");

          if (!default_value.empty())
          {
            auto expressions = Parser::ParseExpressionList(default_value);
            if (expressions.empty())
            {
              throw InternalException("Expression list is empty when parsing default value for column %s", column.name);
            }
            column_def.SetDefaultValue(std::move(expressions[0]));
          }
        }

        info.columns.AddColumn(std::move(column_def));
      }
      arrow_schema.release(&arrow_schema);

      for (auto col_name : not_null_columns)
      {
        auto not_null_index = info.columns.GetColumnIndex(col_name);
        info.constraints.emplace_back(make_uniq<NotNullConstraint>(not_null_index));
      }

      // printf("Creating a table in catalog %s, schema %s, name %s\n", catalog.GetName().c_str(), schema.name.c_str(), info.table.c_str());

      auto table_entry = make_uniq<AirportTableEntry>(catalog, schema, info, row_id_type);
      table_entry->table_data = make_uniq<AirportAPITable>(table);
      CreateEntry(std::move(table_entry));
    }
  }

  optional_ptr<CatalogEntry> AirportTableSet::RefreshTable(ClientContext &context, const string &table_name)
  {
    throw NotImplementedException("AirportTableSet::RefreshTable");
    // auto table_info = GetTableInfo(context, schema, table_name);
    // auto table_entry = make_uniq<AirportTableEntry>(catalog, schema, *table_info);
    // auto table_ptr = table_entry.get();
    // CreateEntry(std::move(table_entry));
    // return table_ptr;
  }

  unique_ptr<AirportTableInfo> AirportTableSet::GetTableInfo(ClientContext &context, AirportSchemaEntry &schema,
                                                             const string &table_name)
  {
    throw NotImplementedException("AirportTableSet::GetTableInfo");
  }

  optional_ptr<CatalogEntry> AirportTableSet::CreateTable(ClientContext &context, BoundCreateTableInfo &info)
  {
    auto &airport_catalog = catalog.Cast<AirportCatalog>();
    auto &base = info.base->Cast<CreateTableInfo>();

    vector<LogicalType> column_types;
    vector<string> column_names;
    for (auto &col : base.columns.Logical())
    {
      column_types.push_back(col.GetType());
      column_names.push_back(col.Name());
    }

    // To perform this creation we likely want to serialize the schema and send it to the server.
    // so the table can be created as part of a DoAction call.

    // So to convert all of the columns an arrow schema we need to look into the code for
    // doing inserts.

    ArrowSchema schema;
    auto client_properties = context.GetClientProperties();

    ArrowConverter::ToArrowSchema(&schema,
                                  column_types,
                                  column_names,
                                  client_properties);

    auto real_schema = arrow::ImportSchema(&schema).ValueOrDie();

    std::shared_ptr<arrow::KeyValueMetadata> schema_metadata = std::make_shared<arrow::KeyValueMetadata>();

    AIRPORT_ARROW_ASSERT_OK_LOCATION(schema_metadata->Set("table_name", base.table), airport_catalog.credentials.location, "");
    AIRPORT_ARROW_ASSERT_OK_LOCATION(schema_metadata->Set("schema_name", base.schema), airport_catalog.credentials.location, "");
    AIRPORT_ARROW_ASSERT_OK_LOCATION(schema_metadata->Set("catalog_name", base.catalog), airport_catalog.credentials.location, "");

    real_schema = real_schema->WithMetadata(schema_metadata);

    // Not make the call, need to include the schema name.

    arrow::flight::FlightCallOptions call_options;

    airport_add_standard_headers(call_options, airport_catalog.credentials.location);

    if (!airport_catalog.credentials.auth_token.empty())
    {
      std::stringstream ss;
      ss << "Bearer " << airport_catalog.credentials.auth_token;
      call_options.headers.emplace_back("authorization", ss.str());
    }
    call_options.headers.emplace_back("airport-action-name", "create_table");

    std::unique_ptr<flight::FlightClient> &flight_client = AirportAPI::FlightClientForLocation(airport_catalog.credentials.location);

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(
        auto serialized_schema,
        arrow::ipc::SerializeSchema(*real_schema, arrow::default_memory_pool()),
        airport_catalog.credentials.location,
        "");

    arrow::flight::Action action{"create_table",
                                 serialized_schema};
    std::unique_ptr<arrow::flight::ResultStream> action_results;
    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(action_results, flight_client->DoAction(call_options, action), airport_catalog.credentials.location, "airport_create_table");

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(auto flight_info_buffer, action_results->Next(), airport_catalog.credentials.location, "");

    if (flight_info_buffer == nullptr)
    {
      throw InternalException("No flight info returned from create_table action");
    }

    std::string_view serialized_flight_info(reinterpret_cast<const char *>(flight_info_buffer->body->data()), flight_info_buffer->body->size());

    // Now how to we deserialize the flight info from that buffer...
    std::shared_ptr<flight::FlightInfo> flight_info;
    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(flight_info, arrow::flight::FlightInfo::Deserialize(serialized_flight_info), airport_catalog.credentials.location, "");

    // We aren't interested in anything after the first result.
    AIRPORT_ARROW_ASSERT_OK_LOCATION(action_results->Drain(), airport_catalog.credentials.location, "");

    // The result is a catalog entry, so we're going to need to create that if the
    // action succeeded.

    auto table_entry = make_uniq<AirportTableEntry>(catalog, this->schema, base, LogicalType::BIGINT);
    AirportAPITable new_table(
        airport_catalog.credentials.location,
        flight_info,
        base.catalog,
        base.schema,
        base.table,
        string(""));

    // Now how are we going to get the flight info, it could just be serialized as part of the action result.
    new_table.flight_info = std::move(flight_info);
    table_entry->table_data = make_uniq<AirportAPITable>(new_table);

    return CreateEntry(std::move(table_entry));
  }

  void AirportTableSet::AlterTable(ClientContext &context, RenameTableInfo &info)
  {
    throw NotImplementedException("AirportTableSet::AlterTable");
  }

  void AirportTableSet::AlterTable(ClientContext &context, RenameColumnInfo &info)
  {
    throw NotImplementedException("AirportTableSet::AlterTable");
  }

  void AirportTableSet::AlterTable(ClientContext &context, AddColumnInfo &info)
  {
    throw NotImplementedException("AirportTableSet::AlterTable");
  }

  void AirportTableSet::AlterTable(ClientContext &context, RemoveColumnInfo &info)
  {
    throw NotImplementedException("AirportTableSet::AlterTable");
  }

  void AirportTableSet::AlterTable(ClientContext &context, AlterTableInfo &alter)
  {
    throw NotImplementedException("AirportTableSet::AlterTable");
  }

  // Given an Arrow schema return a vector of the LogicalTypes for that schema.
  static vector<LogicalType> AirportSchemaToLogicalTypes(
      ClientContext &context,
      std::shared_ptr<arrow::Schema> schema,
      const string &server_location,
      const flight::FlightDescriptor &flight_descriptor)
  {
    ArrowSchemaWrapper schema_root;

    AIRPORT_ARROW_ASSERT_OK_LOCATION_DESCRIPTOR(
        ExportSchema(*schema, &schema_root.arrow_schema),
        server_location,
        flight_descriptor,
        "ExportSchema");

    vector<LogicalType> return_types;

    for (idx_t col_idx = 0;
         col_idx < (idx_t)schema_root.arrow_schema.n_children; col_idx++)
    {
      auto &schema = *schema_root.arrow_schema.children[col_idx];
      if (!schema.release)
      {
        throw InvalidInputException("AirportSchemaToLogicalTypes: released schema passed");
      }
      auto arrow_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), schema);

      if (schema.dictionary)
      {
        auto dictionary_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), *schema.dictionary);
        arrow_type->SetDictionary(std::move(dictionary_type));
      }

      // Indicate that the field should select any type.
      bool is_any_type = false;
      if (schema.metadata != nullptr)
      {
        auto column_metadata = ArrowSchemaMetadata(schema.metadata);
        if (!column_metadata.GetOption("is_any_type").empty())
        {
          is_any_type = true;
        }
      }

      if (is_any_type)
      {
        // This will be sorted out in the bind of the function.
        return_types.push_back(LogicalType::ANY);
      }
      else
      {
        return_types.emplace_back(arrow_type->GetDuckType());
      }
    }
    return return_types;
  }

  void AirportScalarFunctionSet::LoadEntries(ClientContext &context)
  {
    // auto &transaction = AirportTransaction::Get(context, catalog);

    auto &airport_catalog = catalog.Cast<AirportCatalog>();

    // TODO: handle out-of-order columns using position property
    auto curl = connection_pool.acquire();
    auto contents = AirportAPI::GetSchemaItems(
        curl,
        catalog.GetDBPath(),
        schema.name,
        schema.schema_data->contents_url,
        schema.schema_data->contents_sha256,
        schema.schema_data->contents_serialized,
        cache_directory,
        airport_catalog.credentials);

    connection_pool.release(curl);

    //    printf("AirportScalarFunctionSet loading entries\n");
    //    printf("Total functions: %lu\n", tables_and_functions.second.size());

    // There can be functions with the same name.
    std::unordered_map<FunctionCatalogSchemaName, std::vector<AirportAPIScalarFunction>> functions_by_name;

    for (auto &function : contents->scalar_functions)
    {
      FunctionCatalogSchemaName function_key{function.catalog_name, function.schema_name, function.name};
      functions_by_name[function_key].emplace_back(function);
    }

    for (const auto &pair : functions_by_name)
    {
      ScalarFunctionSet flight_func_set(pair.first.name);

      // FIXME: need a way to specify the function stability.
      for (const auto &function : pair.second)
      {
        auto input_types = AirportSchemaToLogicalTypes(context, function.input_schema, function.location, function.flight_info->descriptor());

        arrow::ipc::DictionaryMemo dictionary_memo;

        AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(
            auto output_schema,
            function.flight_info->GetSchema(&dictionary_memo),
            function.location,
            function.flight_info->descriptor(),
            "GetSchema");

        auto output_types = AirportSchemaToLogicalTypes(context, output_schema, function.location, function.flight_info->descriptor());
        D_ASSERT(output_types.size() == 1);

        auto scalar_func = ScalarFunction(input_types, output_types[0],
                                          AirportScalarFun,
                                          AirportScalarFunBind,
                                          nullptr,
                                          nullptr,
                                          AirportScalarFunInitLocalState,
                                          LogicalTypeId::INVALID,
                                          duckdb::FunctionStability::VOLATILE,
                                          duckdb::FunctionNullHandling::DEFAULT_NULL_HANDLING,
                                          nullptr);
        scalar_func.function_info = make_uniq<AirportScalarFunctionInfo>(function.location, function.name, function.flight_info,
                                                                         function.input_schema);

        flight_func_set.AddFunction(scalar_func);
      }

      CreateScalarFunctionInfo info = CreateScalarFunctionInfo(flight_func_set);
      info.catalog = pair.first.catalog_name;
      info.schema = pair.first.schema_name;

      auto function_entry = make_uniq_base<StandardEntry, ScalarFunctionCatalogEntry>(catalog, schema,
                                                                                      info.Cast<CreateScalarFunctionInfo>());
      CreateEntry(std::move(function_entry));
    }
  }

  class AirportDynamicTableFunctionInfo : public TableFunctionInfo
  {
  public:
    std::shared_ptr<AirportAPITableFunction> function;

  public:
    AirportDynamicTableFunctionInfo(const std::shared_ptr<AirportAPITableFunction> function_p)
        : TableFunctionInfo(), function(function_p)
    {
    }

    ~AirportDynamicTableFunctionInfo() override
    {
    }
  };

  static unique_ptr<FunctionData> AirportDynamicTableBind(
      ClientContext &context,
      TableFunctionBindInput &input,
      vector<LogicalType> &return_types,
      vector<string> &names)
  {
    auto function_info = input.info->Cast<AirportDynamicTableFunctionInfo>();

    auto server_location = input.inputs[0].ToString();
    auto params = AirportParseTakeFlightParameters(function_info.function->location,
                                                   context, input);

    // We are going to want these as an arrow appender row and record batch.

    // If we're going to deal with nested types, we don't want to use a msgpack
    // based solution here lets do an Arrow based solution.

    // We need to convert the input schema into a schema that DuckDB can parse into logical types.

    ArrowSchemaWrapper schema_root;
    auto &input_schema = function_info.function->input_schema;

    AIRPORT_ARROW_ASSERT_OK_LOCATION(
        ExportSchema(*input_schema, &schema_root.arrow_schema),
        function_info.function->location,
        "ExportSchema");

    vector<string> input_schema_names;
    vector<LogicalType> input_schema_types;

    for (idx_t col_idx = 0;
         col_idx < (idx_t)schema_root.arrow_schema.n_children; col_idx++)
    {
      auto &schema = *schema_root.arrow_schema.children[col_idx];
      if (!schema.release)
      {
        throw InvalidInputException("airport_dynamic_table_bind: released schema passed");
      }
      auto name = string(schema.name);
      if (name.empty())
      {
        name = string("v") + to_string(col_idx);
      }

      input_schema_names.push_back(name);

      auto arrow_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), schema);
      input_schema_types.push_back(arrow_type->GetDuckType());
    }

    auto appender = make_uniq<ArrowAppender>(input_schema_types, input_schema_types.size(), context.GetClientProperties(),
                                             ArrowTypeExtensionData::GetExtensionTypes(context, input_schema_types));

    // Now we need to make a DataChunk from the input bind data so that we can calle the appender.
    DataChunk input_chunk;
    input_chunk.Initialize(Allocator::Get(context),
                           input_schema_types,
                           1);
    input_chunk.SetCardinality(1);

    // Now how do we populate the input_chunk with the input data?
    int seen_named_parameters = 0;
    for (idx_t col_idx = 0;
         col_idx < (idx_t)schema_root.arrow_schema.n_children; col_idx++)
    {
      auto &schema = *schema_root.arrow_schema.children[col_idx];
      if (!schema.release)
      {
        throw InvalidInputException("airport_dynamic_table_bind: released schema passed");
      }

      // So if the parameter is named, we'd get that off of the metadata
      // otherwise its positional.
      auto metadata = ArrowSchemaMetadata(schema.metadata);
      if (!metadata.GetOption("is_named_parameter").empty())
      {
        input_chunk.data[col_idx].SetValue(0, input.named_parameters[schema.name]);
        seen_named_parameters += 1;
      }
      else
      {
        // Since named parameters aren't passed in inputs, we need to adjust
        // the offset we're looking at.
        auto &input_data = input.inputs[col_idx - seen_named_parameters];
        input_chunk.data[col_idx].SetValue(0, input_data);
      }
    }

    // Now that we have the appender append some data.
    appender->Append(input_chunk, 0, input_chunk.size(), input_chunk.size());
    ArrowArray arr = appender->Finalize();

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(
        auto record_batch,
        arrow::ImportRecordBatch(&arr, function_info.function->input_schema),
        function_info.function->location,
        function_info.function->flight_info->descriptor(), "");

    // Write the record batch to a sequence of bytes.
    // Write RecordBatch to a sequence of bytes

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(auto buffer_output_stream, arrow::io::BufferOutputStream::Create(), function_info.function->location, "create buffer output stream");

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(auto writer, arrow::ipc::MakeStreamWriter(buffer_output_stream, function_info.function->input_schema), function_info.function->location, "make stream writer");

    // Write the RecordBatch
    AIRPORT_ARROW_ASSERT_OK_LOCATION(writer->WriteRecordBatch(*record_batch), function_info.function->location, "write record batch");

    AIRPORT_ARROW_ASSERT_OK_LOCATION(writer->Close(), function_info.function->location, "close record batch writer");

    auto buffer = buffer_output_stream->Finish().ValueOrDie();

    // So save the buffer so we can send it to the server to determine
    // the schema of the flight.

    // Then call the DoAction get_dynamic_flight_info with those arguments.
    GetFlightInfoTableFunctionParameters tf_params;
    tf_params.parameters = string((char *)buffer->data(), buffer->size());
    tf_params.schema_name = function_info.function->schema_name;
    tf_params.action_name = function_info.function->action_name;

    return AirportTakeFlightBindWithFlightDescriptor(
        params,
        function_info.function->flight_info->descriptor(),
        context,
        input, return_types, names, nullptr,
        std::make_shared<GetFlightInfoTableFunctionParameters>(tf_params));
  }

  struct ArrowSchemaTableFunctionTypes
  {
    vector<LogicalType> all;
    vector<string> all_names;
    vector<LogicalType> positional;
    vector<string> positional_names;
    std::map<std::string, LogicalType> named;
  };

  static ArrowSchemaTableFunctionTypes
  AirportSchemaToLogicalTypesWithNaming(
      ClientContext &context,
      std::shared_ptr<arrow::Schema> schema,
      const string &server_location,
      const flight::FlightDescriptor &flight_descriptor)
  {
    ArrowSchemaWrapper schema_root;

    AIRPORT_ARROW_ASSERT_OK_LOCATION_DESCRIPTOR(
        ExportSchema(*schema, &schema_root.arrow_schema),
        server_location,
        flight_descriptor,
        "ExportSchema");

    ArrowSchemaTableFunctionTypes result;

    for (idx_t col_idx = 0;
         col_idx < (idx_t)schema_root.arrow_schema.n_children; col_idx++)
    {
      auto &schema = *schema_root.arrow_schema.children[col_idx];
      if (!schema.release)
      {
        throw InvalidInputException("AirportSchemaToLogicalTypes: released schema passed");
      }
      auto arrow_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), schema);

      if (schema.dictionary)
      {
        auto dictionary_type = ArrowType::GetArrowLogicalType(DBConfig::GetConfig(context), *schema.dictionary);
        arrow_type->SetDictionary(std::move(dictionary_type));
      }

      result.all.emplace_back(arrow_type->GetDuckType());
      result.all_names.emplace_back(string(schema.name));

      auto metadata = ArrowSchemaMetadata(schema.metadata);
      if (!metadata.GetOption("is_named_parameter").empty())
      {
        result.named[schema.name] = arrow_type->GetDuckType();
      }
      else
      {
        result.positional.emplace_back(arrow_type->GetDuckType());
        result.positional_names.push_back(string(schema.name));
      }
    }
    return result;
  }

  void AirportTableFunctionSet::LoadEntries(ClientContext &context)
  {
    // auto &transaction = AirportTransaction::Get(context, catalog);

    auto &airport_catalog = catalog.Cast<AirportCatalog>();

    auto curl = connection_pool.acquire();
    auto contents = AirportAPI::GetSchemaItems(
        curl,
        catalog.GetDBPath(),
        schema.name,
        schema.schema_data->contents_url,
        schema.schema_data->contents_sha256,
        schema.schema_data->contents_serialized,
        cache_directory,
        airport_catalog.credentials);

    connection_pool.release(curl);

    // There can be functions with the same name.
    std::unordered_map<FunctionCatalogSchemaName, std::vector<AirportAPITableFunction>> functions_by_name;

    for (auto &function : contents->table_functions)
    {
      FunctionCatalogSchemaName function_key{function.catalog_name, function.schema_name, function.name};
      functions_by_name[function_key].emplace_back(function);
    }

    for (const auto &pair : functions_by_name)
    {
      TableFunctionSet flight_func_set(pair.first.name);
      vector<FunctionDescription> function_descriptions;

      for (const auto &function : pair.second)
      {
        // These input types are available since they are specified in the metadata, but the
        // schema that is returned likely should be requested dynamically from the dynamic
        // flight function.
        auto input_types = AirportSchemaToLogicalTypesWithNaming(context, function.input_schema, function.location, function.flight_info->descriptor());

        FunctionDescription description;
        description.parameter_types = input_types.positional;
        description.parameter_names = input_types.positional_names;
        description.description = function.description;
        function_descriptions.push_back(std::move(description));

        auto table_func = TableFunction(
            input_types.positional,
            AirportTakeFlight,
            AirportDynamicTableBind,
            AirportArrowScanInitGlobal,
            ArrowTableFunction::ArrowScanInitLocal);

        // Add all of t
        for (auto &named_pair : input_types.named)
        {
          table_func.named_parameters.emplace(named_pair.first, named_pair.second);
        }

        // Need to store some function information along with the function so that when its called
        // we know what to pass to it.
        table_func.function_info = make_uniq<AirportDynamicTableFunctionInfo>(std::make_shared<AirportAPITableFunction>(function));

        flight_func_set.AddFunction(table_func);
      }

      CreateTableFunctionInfo info = CreateTableFunctionInfo(flight_func_set);
      info.catalog = pair.first.catalog_name;
      info.schema = pair.first.schema_name;

      for (auto &desc : function_descriptions)
      {
        info.descriptions.push_back(std::move(desc));
      }

      auto function_entry = make_uniq_base<StandardEntry, TableFunctionCatalogEntry>(catalog, schema,
                                                                                     info.Cast<CreateTableFunctionInfo>());
      CreateEntry(std::move(function_entry));
    }
  }

} // namespace duckdb
