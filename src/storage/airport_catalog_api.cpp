#include <openssl/evp.h>
#include <openssl/sha.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <openssl/sha.h>
#include <curl/curl.h>

#include <arrow/flight/client.h>
#include <arrow/buffer.h>
#include "yyjson.hpp"

#include "storage/airport_catalog_api.hpp"
#include "storage/airport_catalog.hpp"

#include "airport_macros.hpp"
#include "airport_secrets.hpp"
#include "airport_extension.hpp"
#include <curl/curl.h>

namespace flight = arrow::flight;
namespace fs = std::filesystem;
using namespace duckdb_yyjson; // NOLINT

namespace duckdb
{

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

  static void writeToTempFile(const fs::path &tempFilename, const std::string_view &data)
  {
    std::ofstream file(tempFilename, std::ios::binary);
    if (file)
    {
      file.write(data.data(), data.size());
    }
    else
    {
      throw IOException("Airport: Failed to open file for writing: %s", tempFilename.c_str());
    }
  }

  static void renameFile(const fs::path &oldFilename, const fs::path &newFilename)
  {
    std::error_code ec;
    fs::rename(oldFilename, newFilename, ec);
    if (ec)
    {
      throw IOException("Airport: Failed to rename file: %s", ec.message());
    }
  }

  static fs::path generateTempFilename(const fs::path &dir)
  {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 999999);

    fs::path tempFilename;
    do
    {
      tempFilename = dir / ("temp_" + std::to_string(dis(gen)) + ".tmp");
    } while (fs::exists(tempFilename));

    return tempFilename;
  }

  static std::string readFromFile(const fs::path &filename)
  {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
      return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }

  static string TryGetStrFromObject(yyjson_val *obj, const string &field, bool optional = false)
  {
    auto val = yyjson_obj_getn(obj, field.c_str(), field.size());
    if (!val || yyjson_get_type(val) != YYJSON_TYPE_STR)
    {
      if (optional)
      {
        return "";
      }
      throw IOException("JSON parsing error when trying to get field: " + field + " as a string from " + yyjson_val_write(obj, 0, NULL));
    }
    return yyjson_get_str(val);
  }

  vector<string> AirportAPI::GetCatalogs(const string &catalog, AirportCredentials credentials)
  {
    throw NotImplementedException("AirportAPI::GetCatalogs");
  }

  static std::unordered_map<std::string, std::unique_ptr<flight::FlightClient>> airport_flight_clients_by_location;

  static std::unique_ptr<flight::FlightClient> &flightClientForLocation(const std::string &location)
  {
    auto it = airport_flight_clients_by_location.find(location);
    if (it != airport_flight_clients_by_location.end())
    {
      return it->second; // Return a reference to the object
    }

    AIRPORT_ARROW_ASSIGN_OR_RAISE(auto parsed_location,
                                  flight::Location::Parse(location), "()");
    AIRPORT_ARROW_ASSIGN_OR_RAISE(auto created_flight_client, flight::FlightClient::Connect(parsed_location), "(" + location + ")");

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
        throw IOException("Curl Request to '%s' failed with error: '%s'", url, error);
      }
      // Get the HTTP response code
      curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

      if (http_code != 200 or expected_sha256.empty())
      {
        return std::make_pair(http_code, readBuffer);
      }

      // Verify that the SHA256 matches the returned data, don't want a server to
      // corrupt the data.
      auto buffer_view = std::string_view(readBuffer.data(), readBuffer.size());
      auto encountered_sha256 = SHA256ForString(buffer_view);

      if (encountered_sha256 != expected_sha256)
      {
        throw IOException("SHA256 mismatch for URL: %s", url);
      }
      return std::make_pair(http_code, readBuffer);
    }
    throw InternalException("Failed to initialize curl");
  }

  static std::pair<fs::path, fs::path> GetCachePath(const string &input, const string &baseDir)
  {
    fs::path cacheDir = fs::path(baseDir) / "airport_cache"; // Use "cache" subdirectory
    fs::create_directories(cacheDir);                        // Create directory if it doesn't exist

    if (input.size() < 6)
    {
      throw std::invalid_argument("String is too short to contain the SHA256");
    }

    auto subDirName = input.substr(0, 3); // First 3 characters for subdirectory
    auto fileName = input.substr(3);      // Remaining characters for filename

    fs::path subDir = cacheDir / subDirName;
    fs::create_directories(subDir); // Create subdirectory if it doesn't exist

    return std::make_pair(subDir, subDir / fileName);
  }

  static bool fileExists(const fs::path &filePath)
  {
    return fs::exists(filePath) && fs::is_regular_file(filePath);
  }

  void AirportAPI::PopulateURLCacheUsingContainerURL(CURL *curl, const string &url, const string &expected_sha256, const string &baseDir)
  {
    auto sentinel_paths = GetCachePath(expected_sha256, baseDir);

    if (fileExists(sentinel_paths.second))
    {
      // The cache has already been populated.
      return;
    }

    // How do we know if the URLs haven't already been populated.
    auto get_result = GetRequest(curl, url, expected_sha256);

    // We could write a sentinel file.

    if (get_result.first != 200)
    {
      // If this URL cannot be loaded no big deal, just don't populate the cache.
      return;
    }

    // The contents of the file is
    // 4 bytes of the length of the sha256 value
    // the sha 256 value.
    // 4 bytes of the length of the data
    // the data itself.

    auto data = get_result.second.data();

    auto offset = 0;
    const size_t size_of_size = 4;
    while (offset < get_result.second.size())
    {

      // std::string_view serialized_flight_info_length(reinterpret_cast<const char *>(decompressed_url_contents->data()) + offset, 8);

      auto sha256_length = ExtractU32FromString({reinterpret_cast<const char *>(data) + offset, size_of_size});
      offset += size_of_size;

      const std::string sha256_value(reinterpret_cast<const char *>(data) + offset, sha256_length);
      offset += sha256_length;

      auto data_length = ExtractU32FromString({reinterpret_cast<const char *>(data) + offset, size_of_size});
      offset += size_of_size;

      const std::string_view data_value(reinterpret_cast<const char *>(data) + offset, data_length);
      offset += data_length;

      if (SHA256ForString(data_value) != sha256_value)
      {
        // There is corruption.
        throw IOException("SHA256 mismatch from URL: %s for sha256=%s, check for cache corruption", url, sha256_value.c_str());
      }

      auto paths = GetCachePath(sha256_value, baseDir);
      const fs::path subDir = paths.first;
      const fs::path finalFilename = paths.second;

      fs::path tempFilename = generateTempFilename(subDir);
      writeToTempFile(tempFilename, data_value);

      // Rename the temporary file to the final filename
      renameFile(tempFilename, finalFilename);
    }

    // Write a file that the cache has been populated.
    writeToTempFile(sentinel_paths.second, "1");
  }

  // Function to handle caching
  static std::pair<long, std::string> getCachedRequestData(CURL *curl, const string &url, const string &expected_sha256, const fs::path &baseDir)
  {
    if (expected_sha256.empty())
    {
      // Can't cache anything since we don't know the expected sha256 value.
      // and the caching is based on the sha256 values.
      return GetRequest(curl, url, expected_sha256);
    }
    auto paths = GetCachePath(expected_sha256, baseDir);

    const fs::path subDir = paths.first;
    const fs::path finalFilename = paths.second;

    // Check if data is in cache
    std::string cachedData = readFromFile(finalFilename);
    if (!cachedData.empty())
    {
      // Verify that the SHA256 matches the returned data, don't allow a corrupted filesystem
      // to affect things.
      if (!expected_sha256.empty() && SHA256ForString(cachedData) != expected_sha256)
      {
        throw IOException("SHA256 mismatch for URL: %s from cached data at %s, check for cache corruption", url, finalFilename.c_str());
      }
      // printf("Got disk cache hit for %s\n", url.c_str());
      return std::make_pair(200, cachedData);
    }

    // I know this doesn't work for zero byte cached responses, its okay.

    // Data not in cache, fetch it
    auto get_result = GetRequest(curl, url, expected_sha256);

    if (get_result.first != 200)
    {
      return get_result;
    }

    // Save the fetched data to a temporary file
    fs::path tempFilename = generateTempFilename(subDir);
    auto content = std::string_view(get_result.second.data(), get_result.second.size());
    writeToTempFile(tempFilename, content);

    // Rename the temporary file to the final filename
    renameFile(tempFilename, finalFilename);

    // printf("Disk cache miss for %s\n", url.c_str());

    return get_result;
  }

  static void ParseFlightAppMetadata(AirportAPITable &table, const string &catalog, const string &schema)
  {
    auto app_metadata = table.flight_info->app_metadata();

    if (!app_metadata.empty())
    {
      auto doc = yyjson_read(app_metadata.c_str(), app_metadata.size(), 0);
      if (doc)
      {
        auto root = yyjson_doc_get_root(doc);
        if (root && yyjson_get_type(root) == YYJSON_TYPE_OBJ)
        {
          auto object_type = TryGetStrFromObject(root, "type");
          if (object_type == "table")
          {
            table.schema_name = TryGetStrFromObject(root, "schema");
            table.catalog_name = TryGetStrFromObject(root, "catalog");
            table.name = TryGetStrFromObject(root, "name");
            table.comment = TryGetStrFromObject(root, "comment");
          }
        }
        yyjson_doc_free(doc);
      }
    }
  }

  vector<AirportAPITable> AirportAPI::GetTables(CURL *curl,
                                                const string &catalog,
                                                const string &schema,
                                                const string &schema_contents_url,
                                                const string &schema_contents_sha256,
                                                const string &cache_base_dir,
                                                AirportCredentials credentials)
  {
    vector<AirportAPITable> result;
    const string error_location = "(" + credentials.location + ")";

    if (!schema_contents_url.empty())
    {
      auto get_response = getCachedRequestData(curl, schema_contents_url, schema_contents_sha256, cache_base_dir);

      if (get_response.first != 200)
      {
        throw IOException("Failed to get Airport schema contents from URL: %s http response code %ld", schema_contents_url, get_response.first);
      }

      auto url_contents = get_response.second;

      auto url_contents_view = std::string_view(url_contents.data(), url_contents.size());

      auto decompressed_length = ExtractU32FromString(url_contents_view);

      auto codec = arrow::util::Codec::Create(arrow::Compression::ZSTD).ValueOrDie();

      AIRPORT_ARROW_ASSIGN_OR_RAISE(auto decompressed_url_contents, ::arrow::AllocateBuffer(decompressed_length), error_location);

      const size_t size_of_size = 4;

      auto decompress_result = codec->Decompress(
          url_contents.size() - size_of_size, reinterpret_cast<const uint8_t *>(url_contents.data() + size_of_size),
          decompressed_length, decompressed_url_contents->mutable_data());

      AIRPORT_ASSERT_OK(decompress_result, error_location);

      // Now the decompressed result an array of serialized flight::FlightInfo objects, each preceded with the length of the serialization.
      size_t offset = 0;
      const auto *data = decompressed_url_contents->data();
      while (offset < decompressed_length)
      {

        // std::string_view serialized_flight_info_length(reinterpret_cast<const char *>(decompressed_url_contents->data()) + offset, 8);

        auto serialized_length = ExtractU32FromString({reinterpret_cast<const char *>(data) + offset, size_of_size});
        offset += size_of_size;

        std::string_view serialized_flight_info(reinterpret_cast<const char *>(data) + offset, serialized_length);
        offset += serialized_length;

        AIRPORT_ASSIGN_OR_RAISE(auto flight_info, arrow::flight::FlightInfo::Deserialize(serialized_flight_info), error_location);

        AirportAPITable table{
            .location = credentials.location,
            .flight_info = std::move(flight_info)};

        // Look in api_metadata for each flight and determine if it should be a table.
        auto app_metadata = table.flight_info->app_metadata();
        if (!app_metadata.empty())
        {
          ParseFlightAppMetadata(table, catalog, schema);
        }

        if (table.catalog_name == catalog && table.schema_name == schema)
        {
          result.emplace_back(table);
        }
      }
      return result;
    }
    else
    {
      // We need to load the contents of the schemas by listing the flights.
      arrow::flight::FlightCallOptions call_options;
      call_options.headers.emplace_back("airport-user-agent", AIRPORT_USER_AGENT);
      call_options.headers.emplace_back("airport-list-flights-filter-catalog", catalog);
      call_options.headers.emplace_back("airport-list-flights-filter-schema", schema);

      if (!credentials.auth_token.empty())
      {
        std::stringstream ss;
        ss << "Bearer " << credentials.auth_token;
        call_options.headers.emplace_back("authorization", ss.str());
      }

      std::unique_ptr<flight::FlightClient> &flight_client = flightClientForLocation(credentials.location);

      AIRPORT_ARROW_ASSIGN_OR_RAISE(auto listing, flight_client->ListFlights(call_options, {credentials.criteria}), error_location);

      std::unique_ptr<flight::FlightInfo> flight_info;
      AIRPORT_ARROW_ASSIGN_OR_RAISE(flight_info, listing->Next(), error_location);

      while (flight_info != nullptr)
      {
        AirportAPITable table;
        table.location = credentials.location;
        table.flight_info = std::move(flight_info);

        // Look in api_metadata for each flight and determine if it should be a table.
        auto app_metadata = table.flight_info->app_metadata();

        if (!app_metadata.empty())
        {
          ParseFlightAppMetadata(table, catalog, schema);
        }

        if (table.catalog_name == catalog && table.schema_name == schema)
        {
          result.emplace_back(table);
        }

        AIRPORT_ARROW_ASSIGN_OR_RAISE(flight_info, listing->Next(), error_location);
      }

      return result;
    }
  }

  // Function to retrieve a map of strings from a yyjson object
  static unordered_map<string, string> GetMapFromJSON(yyjson_val *root, const char *property_name = nullptr)
  {
    unordered_map<string, string> map;

    // If a property name is provided, look for that property
    if (property_name)
    {
      if (!yyjson_is_obj(root))
      {
        return map;
      }
      root = yyjson_obj_get(root, property_name);
      if (!root || !yyjson_is_obj(root))
      {
        return map;
      }
    }

    // Ensure the root is an object
    if (!yyjson_is_obj(root))
    {
      return map;
    }

    // Iterate over the object key-value pairs
    size_t idx, max;
    yyjson_val *key, *val;
    yyjson_obj_foreach(root, idx, max, key, val)
    {
      // Ensure the key and value are strings
      if (yyjson_is_str(key) && yyjson_is_str(val))
      {
        std::string key_str = yyjson_get_str(key);
        std::string value_str = yyjson_get_str(val);
        map[key_str] = value_str;
      }
    }

    return map;
  }

  // This is not the schemas of the tables.
  unique_ptr<AirportSchemaCollection> AirportAPI::GetSchemas(const string &catalog, AirportCredentials credentials)
  {
    unique_ptr<AirportSchemaCollection> result = make_uniq<AirportSchemaCollection>();

    arrow::flight::FlightCallOptions call_options;

    call_options.headers.emplace_back("airport-user-agent", AIRPORT_USER_AGENT);
    call_options.headers.emplace_back("airport-list-flights-no-schemas", "1");
    call_options.headers.emplace_back("airport-list-flights-listing-schemas", "1");
    call_options.headers.emplace_back("airport-list-flights-filter-catalog", catalog);

    // You need to consult the secret scope.

    if (!credentials.auth_token.empty())
    {
      std::stringstream ss;
      ss << "Bearer " << credentials.auth_token;
      call_options.headers.emplace_back("authorization", ss.str());
    }

    std::unique_ptr<flight::FlightClient> &flight_client = flightClientForLocation(credentials.location);

    arrow::flight::Action action{"list_schemas"};
    std::unique_ptr<arrow::flight::ResultStream> action_results;

    auto error_location_descriptor = "(" + credentials.location + ")";

    AIRPORT_ARROW_ASSIGN_OR_RAISE(action_results, flight_client->DoAction(call_options, action), error_location_descriptor);

    // the first item is the decompressed length
    AIRPORT_ARROW_ASSIGN_OR_RAISE(auto decompressed_schema_length_buffer, action_results->Next(), error_location_descriptor);
    // the second is the compressed schema data.
    AIRPORT_ARROW_ASSIGN_OR_RAISE(auto compressed_schema_data, action_results->Next(), error_location_descriptor);

    // Expand the compressed data that was compressed with zstd.

    auto decompressed_length = ExtractU32FromString({(const char *)decompressed_schema_length_buffer->body->data(), 4});

    auto codec = arrow::util::Codec::Create(arrow::Compression::ZSTD).ValueOrDie();

    AIRPORT_ARROW_ASSIGN_OR_RAISE(auto decompressed_schema_data, ::arrow::AllocateBuffer(decompressed_length), error_location_descriptor);

    auto decompress_result = codec->Decompress(
        compressed_schema_data->body->size(), reinterpret_cast<const uint8_t *>(compressed_schema_data->body->data()),
        decompressed_length, decompressed_schema_data->mutable_data());

    AIRPORT_ASSERT_OK(decompress_result, error_location_descriptor);

    string contents_url;
    string contents_sha256;

    auto doc = yyjson_read((const char *)decompressed_schema_data->data(), decompressed_length, 0);
    if (doc != nullptr)
    {
      auto root = yyjson_doc_get_root(doc);

      auto contents_obj = yyjson_obj_get(root, "contents");
      if (contents_obj && yyjson_is_obj(contents_obj))
      {
        result->contents_url = TryGetStrFromObject(contents_obj, "url");
        result->contents_sha256 = TryGetStrFromObject(contents_obj, "sha256");
      }

      auto schemas_obj = yyjson_obj_get(root, "schemas");
      if (schemas_obj && yyjson_is_arr(schemas_obj))
      {
        size_t idx, max;
        yyjson_val *val;
        yyjson_arr_foreach(schemas_obj, idx, max, val)
        {
          if (yyjson_is_obj(val))
          {
            auto schema_name = TryGetStrFromObject(val, "schema");
            auto description = TryGetStrFromObject(val, "description");

            auto tags = GetMapFromJSON(val, "tags");

            AirportAPISchema schema_result;
            schema_result.schema_name = TryGetStrFromObject(val, "schema");
            schema_result.catalog_name = catalog;
            schema_result.comment = TryGetStrFromObject(val, "description", true);
            schema_result.tags = GetMapFromJSON(val, "tags");

            auto contents_obj = yyjson_obj_get(val, "contents");
            if (contents_obj && yyjson_is_obj(contents_obj))
            {
              schema_result.contents_url = TryGetStrFromObject(contents_obj, "url");
              schema_result.contents_sha256 = TryGetStrFromObject(contents_obj, "sha256");
            }

            result->schemas.emplace_back(schema_result);
          }
        }
      }
    }
    else
    {
      throw IOException("Failed to parse JSON document describing available Arrow Flight schemas");
    }
    yyjson_doc_free(doc);

    AIRPORT_ARROW_ASSERT_OK(action_results->Drain(), error_location_descriptor);

    return result;
  }

} // namespace duckdb
