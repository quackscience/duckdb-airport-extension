#include <openssl/evp.h>
#include <openssl/sha.h>
#include <iomanip>
#include <random>
#include <string_view>
#include <vector>
#include <openssl/sha.h>
#include <curl/curl.h>

#include <arrow/flight/client.h>
#include <arrow/buffer.h>
#include "yyjson.hpp"

#include "storage/airport_catalog_api.hpp"
#include "storage/airport_catalog.hpp"

#include "duckdb/common/file_system.hpp"

#include "airport_macros.hpp"
#include "airport_secrets.hpp"
#include "airport_headers.hpp"
#include "airport_extension.hpp"
#include <curl/curl.h>
#include <msgpack.hpp>
#include <arrow/ipc/api.h>
#include <arrow/io/memory.h>

namespace flight = arrow::flight;
using namespace duckdb_yyjson; // NOLINT

namespace duckdb
{

  static constexpr idx_t FILE_FLAGS_READ = idx_t(1 << 0);
  static constexpr idx_t FILE_FLAGS_WRITE = idx_t(1 << 1);
  //  static constexpr idx_t FILE_FLAGS_FILE_CREATE = idx_t(1 << 3);
  static constexpr idx_t FILE_FLAGS_FILE_CREATE_NEW = idx_t(1 << 4);

  inline static uint32_t ExtractU32FromString(std::string_view str)
  {
    if (str.size() < 4)
    {
      throw std::invalid_argument("String is too short to contain four bytes.");
    }

    uint64_t result = 0;
    std::copy(str.data(), str.data() + 4, reinterpret_cast<char *>(&result));

    return result;
  }

  static void writeToTempFile(FileSystem &fs, const string &tempFilename, const std::string_view &data)
  {
    auto handle = fs.OpenFile(tempFilename, FILE_FLAGS_WRITE | FILE_FLAGS_FILE_CREATE_NEW);
    if (!handle)
    {
      throw IOException("Airport: Failed to open file for writing: %s", tempFilename.c_str());
    }

    handle->Write((void *)data.data(), data.size());
    handle->Sync();
    handle->Close();
  }

  static string generateTempFilename(FileSystem &fs, const string &dir)
  {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 999999);

    string filename;

    do
    {
      filename = fs.JoinPath(dir, "temp_" + std::to_string(dis(gen)) + ".tmp");
    } while (fs.FileExists(filename));
    return filename;
  }

  static std::string readFromFile(FileSystem &fs, const string &filename)
  {
    auto handle = fs.OpenFile(filename, FILE_FLAGS_READ);
    if (!handle)
    {
      return "";
    }
    auto file_size = handle->GetFileSize();
    string read_buffer = string(file_size, '\0');
    handle->Read((void *)read_buffer.data(), file_size);
    return read_buffer;
  }

  vector<string> AirportAPI::GetCatalogs(const string &catalog, AirportCredentials credentials)
  {
    throw NotImplementedException("AirportAPI::GetCatalogs");
  }

  static std::unordered_map<std::string, std::unique_ptr<flight::FlightClient>> airport_flight_clients_by_location;

  std::unique_ptr<flight::FlightClient> &AirportAPI::FlightClientForLocation(const std::string &location)
  {
    auto it = airport_flight_clients_by_location.find(location);
    if (it != airport_flight_clients_by_location.end())
    {
      return it->second; // Return a reference to the object
    }

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(auto parsed_location,
                                            flight::Location::Parse(location), location, "");
    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(auto created_flight_client, flight::FlightClient::Connect(parsed_location), location, "");

    airport_flight_clients_by_location[location] = std::move(created_flight_client);

    return airport_flight_clients_by_location[location];
  }

  static size_t GetRequestWriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
  {
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
  }

  static std::string SHA256ForString(const std::string_view &input)
  {
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    const EVP_MD *md = EVP_sha256();

    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int lengthOfHash = 0;

    EVP_DigestInit_ex(context, md, nullptr);
    EVP_DigestUpdate(context, input.data(), input.size());
    EVP_DigestFinal_ex(context, hash, &lengthOfHash);
    EVP_MD_CTX_free(context);

    std::stringstream ss;
    for (unsigned int i = 0; i < lengthOfHash; ++i)
    {
      ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }

    return ss.str();
  }

  static std::string SHA256ForString(const std::string &input)
  {
    EVP_MD_CTX *context = EVP_MD_CTX_new();
    const EVP_MD *md = EVP_sha256();

    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int lengthOfHash = 0;

    EVP_DigestInit_ex(context, md, nullptr);
    EVP_DigestUpdate(context, input.data(), input.size());
    EVP_DigestFinal_ex(context, hash, &lengthOfHash);
    EVP_MD_CTX_free(context);

    std::stringstream ss;
    for (unsigned int i = 0; i < lengthOfHash; ++i)
    {
      ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }

    return ss.str();
  }

  static std::pair<long, std::string> GetRequest(CURL *curl, const string &url, const string expected_sha256)
  {
    CURLcode res;
    string readBuffer;
    long http_code = 0;

    if (curl)
    {
      // Enable HTTP/2
      curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2);
      curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, GetRequestWriteCallback);
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
      res = curl_easy_perform(curl);

      if (res != CURLcode::CURLE_OK)
      {
        string error = curl_easy_strerror(res);
        throw IOException("Curl Request to " + url + " failed with error: " + error);
      }
      // Get the HTTP response code
      curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

      if (http_code != 200 || expected_sha256.empty())
      {
        return std::make_pair(http_code, readBuffer);
      }

      // Verify that the SHA256 matches the returned data, don't want a server to
      // corrupt the data.
      auto buffer_view = std::string_view(readBuffer.data(), readBuffer.size());
      auto encountered_sha256 = SHA256ForString(buffer_view);

      if (encountered_sha256 != expected_sha256)
      {
        throw IOException("SHA256 mismatch for URL: " + url);
      }
      return std::make_pair(http_code, readBuffer);
    }
    throw InternalException("Failed to initialize curl");
  }

  static std::pair<const string, const string> GetCachePath(FileSystem &fs, const string &input, const string &baseDir)
  {
    auto cacheDir = fs.JoinPath(baseDir, "airport_cache");
    if (!fs.DirectoryExists(cacheDir))
    {
      fs.CreateDirectory(cacheDir);
    }

    if (input.size() < 6)
    {
      throw std::invalid_argument("String is too short to contain the SHA256");
    }

    auto subDirName = input.substr(0, 3); // First 3 characters for subdirectory
    auto fileName = input.substr(3);      // Remaining characters for filename

    auto subDir = fs.JoinPath(cacheDir, subDirName);
    if (!fs.DirectoryExists(subDir))
    {
      fs.CreateDirectory(subDir);
    }

    return std::make_pair(subDir, fs.JoinPath(subDir, fileName));
  }

  void
  AirportAPI::PopulateCatalogSchemaCacheFromURLorContent(CURL *curl,
                                                         const AirportSchemaCollection &collection,
                                                         const string &catalog_name,
                                                         const string &baseDir)
  {
    auto fs = FileSystem::CreateLocal();

    if (collection.contents_sha256.empty())
    {
      // If the collection doesn't have a SHA256 there isn't anything we can do.
      return;
    }

    // If there is a large amount of data to be populate its useful to determine
    // if the SHA256 value of all schemas has already been populated.
    //
    // So we have a sentinel path that indicates the entire contents of the SHA256
    // has already been written out to the cache.
    auto sentinel_paths = GetCachePath(*fs, collection.contents_sha256, baseDir);

    if (fs->FileExists(sentinel_paths.second))
    {
      // All of the schemas for this overall SHA256
      // have been populated so there is nothing to do.
      return;
    }

    // The schema data is serialized using msgpack
    //
    // it can either be retrieved from a URL or provided inline.
    //
    // The schemas are packed like:
    // [
    //   [string, string]
    // ]
    // the first item is the SHA256 of the data, then the actual data
    // is the second item.
    //
    msgpack::object_handle oh;
    if (!collection.contents_serialized.empty())
    {
      const string found_sha = SHA256ForString(collection.contents_serialized);
      if (found_sha != collection.contents_sha256)
      {
        // Can't use the serrialized data since the SHA256 doesn't match, so
        // just give up.
        //
        // Don't raise an error right now.
        return;
      }
      oh = msgpack::unpack((const char *)collection.contents_serialized.data(), collection.contents_serialized.size(), 0);
    }
    else
    {
      // How do we know if the URLs haven't already been populated.
      auto get_result = GetRequest(curl, collection.schema_collection_contents_url, collection.contents_sha256);

      if (get_result.first != 200)
      {
        // If this URL cannot be loaded no big deal, just don't populate the cache, don't raise
        // an error right now.
        return;
      }
      oh = msgpack::unpack((const char *)get_result.second.data(), get_result.second.size(), 0);
    }

    std::vector<std::vector<std::string>> unpacked_data = oh.get().as<std::vector<std::vector<std::string>>>();

    for (auto &item : unpacked_data)
    {
      if (item.size() != 2)
      {
        throw IOException("Catalog schema cache contents had an item where size != 2");
      }

      const string found_sha = SHA256ForString(item[1]);
      const string &expected_sha = item[0];
      if (found_sha != expected_sha)
      {
        auto error_prefix = "Catalog " + catalog_name + " SHA256 Mismatch expected " + collection.contents_sha256 + " found " + found_sha;

        // There is corruption.
        if (collection.contents_serialized.empty())
        {
          throw IOException(error_prefix + " from URL: " + collection.schema_collection_contents_url);
        }
        else
        {
          throw IOException(error_prefix + " from serialized content");
        }
      }

      auto paths = GetCachePath(*fs, item[0], baseDir);
      auto tempFilename = generateTempFilename(*fs, paths.first);

      writeToTempFile(*fs, tempFilename, item[1]);

      // Rename the temporary file to the final filename
      fs->MoveFile(tempFilename, paths.second);
    }

    // Write a file that the cache has been populated with the top level SHA256
    // value, so that we can skip doing this next time all schemas are used.
    writeToTempFile(*fs, sentinel_paths.second, "1");
  }

  // Function to handle caching
  static std::pair<long, std::string> getCachedRequestData(CURL *curl, const string &url,
                                                           const string &expected_sha256,
                                                           const string &serialized,
                                                           const string &baseDir)
  {
    if (expected_sha256.empty())
    {
      // Can't cache anything since we don't know the expected sha256 value.
      // and the caching is based on the sha256 values.
      //
      // So if there was inline content supplied use that and fake that it was
      // retrieved from a server.
      if (!serialized.empty())
      {
        return std::make_pair(200, serialized);
      }
      else
      {
        return GetRequest(curl, url, expected_sha256);
      }
    }

    // If the user supplied an inline serialized value, check if the sha256 matches, if so
    // use it otherwise fall abck to the url
    if (!serialized.empty())
    {
      if (SHA256ForString(serialized) == expected_sha256)
      {
        return std::make_pair(200, serialized);
      }
      if (url.empty())
      {
        throw IOException("SHA256 mismatch inline serialized data and URL is empty");
      }
    }

    auto fs = FileSystem::CreateLocal();

    auto paths = GetCachePath(*fs, expected_sha256, baseDir);

    // Check if data is in cache
    if (fs->FileExists(paths.second))
    {
      std::string cachedData = readFromFile(*fs, paths.second);
      if (!cachedData.empty())
      {
        // Verify that the SHA256 matches the returned data, don't allow a corrupted filesystem
        // to affect things.
        if (!expected_sha256.empty() && SHA256ForString(cachedData) != expected_sha256)
        {
          throw IOException("SHA256 mismatch for URL: %s from cached data at %s, check for cache corruption", url, paths.second.c_str());
        }
        return std::make_pair(200, cachedData);
      }
    }

    // I know this doesn't work for zero byte cached responses, its okay.

    // Data not in cache, fetch it
    auto get_result = GetRequest(curl, url, expected_sha256);

    if (get_result.first != 200)
    {
      return get_result;
    }

    // Save the fetched data to a temporary file
    auto tempFilename = generateTempFilename(*fs, paths.first);
    auto content = std::string_view(get_result.second.data(), get_result.second.size());
    writeToTempFile(*fs, tempFilename, content);

    // Rename the temporary file to the final filename
    fs->MoveFile(tempFilename, paths.second);
    return get_result;
  }

  struct SerializedFlightAppMetadata
  {
    // This will either be a "table" or "function"
    //
    // table is a sql table
    // function is a scalar function.
    string type;
    string schema;
    string catalog;
    string name;
    string comment;

    // This is the Arrow serialized schema for the input to the function, its not set
    // on tables.
    std::optional<string> input_schema;
    std::optional<string> action_name;

    // This is the function description for table or scalar functions.
    std::optional<string> description;

    MSGPACK_DEFINE_MAP(type, schema, catalog, name, comment, input_schema, action_name, description);
  };

  static std::unique_ptr<SerializedFlightAppMetadata> ParseFlightAppMetadata(const string &app_metadata)
  {
    SerializedFlightAppMetadata app_metadata_obj;
    try
    {
      msgpack::object_handle oh = msgpack::unpack((const char *)app_metadata.data(), app_metadata.size(), 0);
      msgpack::object obj = oh.get();
      obj.convert(app_metadata_obj);
    }
    catch (const std::exception &e)
    {
      throw InvalidInputException("File to parse MsgPack object describing in Arrow Flight app_metadata %s", e.what());
    }
    return std::make_unique<SerializedFlightAppMetadata>(app_metadata_obj);
  }

  void handle_flight_app_metadata(const string &app_metadata,
                                  const string &target_catalog,
                                  const string &target_schema,
                                  const string &location,
                                  const std::shared_ptr<arrow::flight::FlightInfo> &flight_info,
                                  unique_ptr<AirportSchemaContents> &contents)
  {
    auto parsed_app_metadata = ParseFlightAppMetadata(app_metadata);
    if (!(parsed_app_metadata->catalog == target_catalog && parsed_app_metadata->schema == target_schema))
    {
      return;
    }

    if (parsed_app_metadata->type == "table")
    {
      AirportAPITable table(
          location,
          std::move(flight_info),
          parsed_app_metadata->catalog,
          parsed_app_metadata->schema,
          parsed_app_metadata->name,
          parsed_app_metadata->comment);
      contents->tables.emplace_back(table);
    }
    else if (parsed_app_metadata->type == "table_function")
    {
      AirportAPITableFunction function{
          .location = location,
          .flight_info = flight_info,
          .catalog_name = parsed_app_metadata->catalog,
          .schema_name = parsed_app_metadata->schema,
          .name = parsed_app_metadata->name,
          .comment = parsed_app_metadata->comment,
          .action_name = parsed_app_metadata->action_name.value_or(""),
          .description = parsed_app_metadata->description.value_or("")};

      if (!parsed_app_metadata->input_schema.has_value())
      {
        throw IOException("Table Function metadata does not have an input_schema defined for function " + function.schema_name + "." + function.name);
      }

      auto serialized_schema = parsed_app_metadata->input_schema.value();

      arrow::io::BufferReader parameter_schema_reader(
          std::make_shared<arrow::Buffer>(serialized_schema));

      arrow::ipc::DictionaryMemo in_memo;
      AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(
          auto parameter_schema,
          arrow::ipc::ReadSchema(&parameter_schema_reader, &in_memo),
          location,
          function.flight_info->descriptor(),
          "Read serialized input schema");

      function.input_schema = parameter_schema;

      contents->table_functions.emplace_back(function);
    }
    else if (parsed_app_metadata->type == "scalar_function")
    {
      AirportAPIScalarFunction function{
          .location = location,
          .flight_info = std::move(flight_info)};

      function.catalog_name = parsed_app_metadata->catalog;
      function.schema_name = parsed_app_metadata->schema;
      function.name = parsed_app_metadata->name;
      function.comment = parsed_app_metadata->comment;
      function.description = parsed_app_metadata->description.value_or("");

      if (!parsed_app_metadata->input_schema.has_value())
      {
        throw IOException("Function metadata does not have an input_schema defined for function " + function.schema_name + "." + function.name);
      }

      auto serialized_schema = parsed_app_metadata->input_schema.value();

      arrow::io::BufferReader parameter_schema_reader(
          std::make_shared<arrow::Buffer>(serialized_schema));

      arrow::ipc::DictionaryMemo in_memo;
      AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION_DESCRIPTOR(
          auto parameter_schema,
          arrow::ipc::ReadSchema(&parameter_schema_reader, &in_memo),
          location,
          function.flight_info->descriptor(),
          "Read serialized input schema");

      function.input_schema = parameter_schema;

      contents->scalar_functions.emplace_back(function);
    }
    else
    {
      throw IOException("Unknown object type in app_metadata: " + parsed_app_metadata->type);
    }
  }

  //  std::pair<vector<AirportAPITable>, vector<AirportAPIScalarFunction>>
  unique_ptr<AirportSchemaContents>
  AirportAPI::GetSchemaItems(CURL *curl,
                             const string &catalog,
                             const string &schema,
                             const string &schema_contents_url,
                             const string &schema_contents_sha256,
                             const string &schema_contents_serialized,
                             const string &cache_base_dir,
                             AirportCredentials credentials)
  {
    auto contents = make_uniq<AirportSchemaContents>();

    if (!(schema_contents_url.empty() && schema_contents_serialized.empty() && schema_contents_sha256.empty()))
    {
      string url_contents;
      auto get_response = getCachedRequestData(curl, schema_contents_url, schema_contents_sha256, schema_contents_serialized, cache_base_dir);

      if (get_response.first != 200)
      {
        throw IOException("Failed to get Airport schema contents from URL: %s http response code %ld", schema_contents_url, get_response.first);
      }
      url_contents = get_response.second;

      // So this data has this layout

      // We should change this to message pack.

      // 4 byte length - decompressed length of contents
      //
      // Then the decompressed result is just a list
      // of serialized FlightInfo objects.
      auto url_contents_view = std::string_view(url_contents.data(), url_contents.size());

      auto decompressed_length = ExtractU32FromString(url_contents_view);
      auto codec = arrow::util::Codec::Create(arrow::Compression::ZSTD).ValueOrDie();
      AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(auto decompressed_url_contents, ::arrow::AllocateBuffer(decompressed_length), credentials.location, "");

      const size_t size_of_size = 4;
      auto decompress_result = codec->Decompress(
          url_contents.size() - size_of_size, reinterpret_cast<const uint8_t *>(url_contents.data() + size_of_size),
          decompressed_length, decompressed_url_contents->mutable_data());

      AIRPORT_ARROW_ASSERT_OK_LOCATION(decompress_result, credentials.location, "");

      msgpack::object_handle oh = msgpack::unpack((const char *)decompressed_url_contents->data(), decompressed_length, 0);
      std::vector<std::string> unpacked_data = oh.get().as<std::vector<std::string>>();

      for (auto item : unpacked_data)
      {
        AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(auto flight_info, arrow::flight::FlightInfo::Deserialize(item), credentials.location, "");

        // Look in api_metadata for each flight and determine if it should be a table.
        auto app_metadata = flight_info->app_metadata();
        if (app_metadata.empty())
        {
          continue;
        }
        handle_flight_app_metadata(app_metadata, catalog, schema, credentials.location, std::move(flight_info), contents);
      }

      return contents;
    }
    else
    {
      // We need to load the contents of the schemas by listing the flights.
      arrow::flight::FlightCallOptions call_options;
      airport_add_standard_headers(call_options, credentials.location);
      call_options.headers.emplace_back("airport-list-flights-filter-catalog", catalog);
      call_options.headers.emplace_back("airport-list-flights-filter-schema", schema);

      if (!credentials.auth_token.empty())
      {
        std::stringstream ss;
        ss << "Bearer " << credentials.auth_token;
        call_options.headers.emplace_back("authorization", ss.str());
      }

      std::unique_ptr<flight::FlightClient> &flight_client = FlightClientForLocation(credentials.location);

      AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(auto listing, flight_client->ListFlights(call_options, {credentials.criteria}), credentials.location, "");

      std::shared_ptr<flight::FlightInfo> flight_info;
      AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(flight_info, listing->Next(), credentials.location, "");

      while (flight_info != nullptr)
      {
        // Look in api_metadata for each flight and determine if it should be a table.
        auto app_metadata = flight_info->app_metadata();
        if (app_metadata.empty())
        {
          continue;
        }
        handle_flight_app_metadata(app_metadata, catalog, schema, credentials.location, flight_info, contents);

        // FIXME: there needs to be more code here to create the actual table.
        // Rusty look into this.

        AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(flight_info, listing->Next(), credentials.location, "");
      }

      return contents;
    }
  }

  static std::string create_schema_request_document(const string &catalog)
  {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);

    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_str(doc, root, "catalog_name", catalog.c_str());

    char *json_str = yyjson_mut_write(doc, 0, nullptr);
    std::string result(json_str);

    free(json_str);
    yyjson_mut_doc_free(doc);

    return result;
  }

  struct SerializedContents
  {
    std::optional<std::string> url;
    std::optional<std::string> sha256;
    std::optional<std::string> serialized;
    MSGPACK_DEFINE_MAP(url, sha256, serialized);
  };

  struct SerializedSchema
  {
    std::string schema;
    std::string description;
    std::unordered_map<std::string, std::string> tags;
    SerializedContents contents;

    MSGPACK_DEFINE_MAP(schema, description, tags, contents);
  };

  struct SerializedCatalogRoot
  {
    SerializedContents contents;
    std::vector<SerializedSchema> schemas;

    MSGPACK_DEFINE_MAP(contents, schemas); // MessagePack mapping
  };

  // This is not the schemas of the tables.
  unique_ptr<AirportSchemaCollection>
  AirportAPI::GetSchemas(const string &catalog, AirportCredentials credentials)
  {
    unique_ptr<AirportSchemaCollection> result = make_uniq<AirportSchemaCollection>();

    arrow::flight::FlightCallOptions call_options;

    airport_add_standard_headers(call_options, credentials.location);
    //    call_options.headers.emplace_back("airport-list-flights-no-schemas", "1");
    //    call_options.headers.emplace_back("airport-list-flights-listing-schemas", "1");
    //    call_options.headers.emplace_back("airport-list-flights-filter-catalog", catalog);

    if (!credentials.auth_token.empty())
    {
      std::stringstream ss;
      ss << "Bearer " << credentials.auth_token;
      call_options.headers.emplace_back("authorization", ss.str());
    }
    call_options.headers.emplace_back("airport-action-name", "list_schemas");

    std::unique_ptr<flight::FlightClient> &flight_client = FlightClientForLocation(credentials.location);

    arrow::flight::Action action{"list_schemas", arrow::Buffer::FromString(create_schema_request_document(catalog))};
    std::unique_ptr<arrow::flight::ResultStream> action_results;
    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(action_results, flight_client->DoAction(call_options, action), credentials.location, "");

    // the first item is the decompressed length
    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(auto decompressed_schema_length_buffer, action_results->Next(), credentials.location, "");

    if (decompressed_schema_length_buffer == nullptr)
    {
      throw AirportFlightException(credentials.location, "Failed to obtain schema data from Arrow Flight server via DoAction()");
    }
    // the second is the compressed schema data.
    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(auto compressed_schema_data, action_results->Next(), credentials.location, "");

    // Expand the compressed data that was compressed with zstd.

    auto decompressed_length = ExtractU32FromString({(const char *)decompressed_schema_length_buffer->body->data(), 4});

    auto codec = arrow::util::Codec::Create(arrow::Compression::ZSTD).ValueOrDie();

    AIRPORT_FLIGHT_ASSIGN_OR_RAISE_LOCATION(auto decompressed_schema_data, ::arrow::AllocateBuffer(decompressed_length), credentials.location, "");

    auto decompress_result = codec->Decompress(
        compressed_schema_data->body->size(), reinterpret_cast<const uint8_t *>(compressed_schema_data->body->data()),
        decompressed_length, decompressed_schema_data->mutable_data());

    AIRPORT_ARROW_ASSERT_OK_LOCATION(decompress_result, credentials.location, "");

    SerializedCatalogRoot catalog_root;
    try
    {
      msgpack::object_handle oh = msgpack::unpack((const char *)decompressed_schema_data->data(), decompressed_length, 0);
      msgpack::object obj = oh.get();
      obj.convert(catalog_root);
    }
    catch (const std::exception &e)
    {
      throw InvalidInputException("File to parse MsgPack object describing catalog root %s", e.what());
    }

    result->schema_collection_contents_url = catalog_root.contents.url ? catalog_root.contents.url.value() : "";
    result->contents_sha256 = catalog_root.contents.sha256 ? catalog_root.contents.sha256.value() : "";
    result->contents_serialized = catalog_root.contents.serialized ? catalog_root.contents.serialized.value() : "";

    for (auto &schema : catalog_root.schemas)
    {
      AirportAPISchema schema_result;
      schema_result.schema_name = schema.schema;
      schema_result.catalog_name = catalog;
      schema_result.comment = schema.description;
      schema_result.tags = schema.tags;
      schema_result.contents_url = schema.contents.url ? schema.contents.url.value() : "";
      schema_result.contents_sha256 = schema.contents.sha256 ? schema.contents.sha256.value() : "";
      schema_result.contents_serialized = schema.contents.serialized ? schema.contents.serialized.value() : "";
      result->schemas.emplace_back(schema_result);
    }

    AIRPORT_ARROW_ASSERT_OK_LOCATION(action_results->Drain(), credentials.location, "");

    return result;
  }

} // namespace duckdb
