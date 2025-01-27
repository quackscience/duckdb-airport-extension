#pragma once

#include "duckdb/common/types.hpp"

#include <arrow/flight/client.h>
#include <curl/curl.h>

namespace duckdb
{
  struct AirportCredentials;

  struct AirportAPITable
  {
    string location;
    std::shared_ptr<arrow::flight::FlightInfo> flight_info;

    string catalog_name;
    string schema_name;
    string name;
    string comment;

    AirportAPITable(
        const std::string &location,
        std::shared_ptr<arrow::flight::FlightInfo> flightInfo,
        const std::string &catalog,
        const std::string &schema,
        const std::string &tableName,
        const std::string &tableComment)
        : location(location),
          flight_info(flightInfo),
          catalog_name(catalog),
          schema_name(schema),
          name(tableName),
          comment(tableComment) {}
  };

  struct AirportAPIScalarFunction
  {
    string catalog_name;
    string schema_name;
    string name;

    string comment;
    string description;

    string location;
    std::shared_ptr<arrow::flight::FlightInfo> flight_info;
    std::shared_ptr<arrow::Schema> input_schema;
  };

  struct AirportAPITableFunction
  {
    string catalog_name;
    string schema_name;

    // The name of the table function.
    string name;
    string description;
    string comment;

    // The name of the action passed, if there is a single
    // flight that exists it can respond with different outputs
    // based on this name.
    string action_name;

    // The location of the flight server that will prduce the data.
    string location;

    // This is the flight that will be called to satisfy the function.
    std::shared_ptr<arrow::flight::FlightInfo> flight_info;

    // The schema of the input to the function.
    std::shared_ptr<arrow::Schema> input_schema;
  };

  struct AirportAPISchema
  {
    string schema_name;
    string catalog_name;
    string comment;
    unordered_map<string, string> tags;

    // An optional URL to the contents of the schema.
    string contents_url;
    // The SHA256 hash of the contents of the schema.
    string contents_sha256;

    // The actual contents of the schema if provided inline.
    string contents_serialized;
  };

  struct AirportSchemaCollection
  {
    // An optional URL that contains all of the contents for all of the schemas.
    string schema_collection_contents_url;
    // The SHA256 of the contents url.
    string contents_sha256;
    string contents_serialized;

    vector<AirportAPISchema> schemas;
  };

  // A collection of parsed items from a schema's metadata.
  struct AirportSchemaContents
  {
  public:
    vector<AirportAPITable> tables;
    vector<AirportAPIScalarFunction> scalar_functions;
    vector<AirportAPITableFunction> table_functions;
  };

  class AirportAPI
  {
  public:
    static vector<string> GetCatalogs(const string &catalog, AirportCredentials credentials);
    static unique_ptr<AirportSchemaContents> GetSchemaItems(CURL *curl,
                                                            const string &catalog,
                                                            const string &schema,
                                                            const string &schema_contents_url,
                                                            const string &schema_contents_sha256,
                                                            const string &schema_contents_serialized,
                                                            const string &cache_base_dir,
                                                            AirportCredentials credentials);
    static unique_ptr<AirportSchemaCollection> GetSchemas(const string &catalog, AirportCredentials credentials);

    static void PopulateCatalogSchemaCacheFromURLorContent(CURL *curl,
                                                           const AirportSchemaCollection &collection,
                                                           const string &catalog_name,
                                                           const string &baseDir);

    static std::unique_ptr<arrow::flight::FlightClient> &FlightClientForLocation(const std::string &location);
  };

} // namespace duckdb
