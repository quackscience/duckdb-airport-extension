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

struct TakeFlightParameters {
    string server_location;
    string auth_token;
    string secret_name;
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

static unique_ptr<FunctionData> take_flight_custom_ticket_bind(
    ClientContext &context,
    TableFunctionBindInput &input,
    vector<LogicalType> &return_types,
    vector<string> &names) {
    TakeFlightParameters params = ParseTakeFlightParameters(input.inputs[0].ToString(), context, input);
    auto custom_ticket = input.inputs[1].ToString();

    flight::FlightDescriptor descriptor = flight_custom_ticket_descriptor(custom_ticket);

    return take_flight_bind_with_descriptor(params, descriptor, context, input, return_types, names, nullptr);
}

void AddTakeFlightCustomFunction(DatabaseInstance &instance) {
    auto take_flight_custom_function_set = TableFunctionSet("airport_take_flight_custom");

    auto take_flight_custom_function = TableFunction(
        "airport_take_flight_custom",
        {LogicalType::VARCHAR, LogicalType::BLOB},
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
