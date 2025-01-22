#pragma once

#include "duckdb.hpp"


namespace duckdb
{

	class AirportExtension : public Extension
	{
	public:
		void Load(DuckDB &db) override;
		std::string Name() override;
		std::string Version() const override;
	};

	void AddListFlightsFunction(DatabaseInstance &instance);

	void AddTakeFlightFunction(DatabaseInstance &instance);

        void AddTakeCustomFlightFunction(DatabaseInstance &instance);

} // namespace duckdb
