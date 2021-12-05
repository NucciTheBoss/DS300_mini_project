#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sqlite3
from flask import Flask, jsonify
from typing import Any, List, Union


# Global variables
app = Flask(__name__)
DF = None


def parse(query: str) -> str:
    """
    Parse URL query into correct SQL syntax.

    :param query: SQL query pulled from URL argument.
    :return: Parsed query converted to valid SQL syntax.
    """
    query_split = query.split("+")
    return " ".join(query_split)


class DBManager:
    """
    Class to handle communication with SQLite database.
    """

    _class_instance = None
    _db_instance = False
    _connection = None
    _cursor = None

    def __init__(self) -> None:
        """ Private constructor. """
        if DBManager._class_instance is not None:
            raise Exception("This class is a singleton.")

        else:
            DBManager._class_instance = self

    @staticmethod
    def get_instance():
        """
        Retrieve instance of DBManager.

        :return: Instance of DBManager class.
        """
        if DBManager._class_instance is None:
            DBManager()

        return DBManager._class_instance

    @staticmethod
    def create_db(path_to_csv: str) -> None:
        """
        Create ephemeral SQL database in RAM. Database goes away after program ends.

        :param path_to_csv: Path to CSV file.
        """
        if DBManager._db_instance is False:
            # Create DB in RAM and cursor to navigate database
            DBManager._connection = sqlite3.connect(":memory:", check_same_thread=False)
            DBManager._cursor = DBManager._connection.cursor()

            # Read in csv file and add to the DB
            driving_data = pd.read_csv(path_to_csv)
            driving_data.to_sql("PPP", DBManager._connection, if_exists="replace")

            # Set to true so that we do not run block more than once
            DBManager._db_instance = True

    def execute(self, query: str) -> Any:
        """
        Execute query on SQLite DB.

        :param query: SQL query pulled from URL argument.
        :return: ResultSet or float/int/str value.
        """
        return self._cursor.execute(parse(query)).fetchall()


class DFManager:
    """
    Class to manage differential privacy mechanism and access to database.

    Assumptions:
    - Using laplace mechanism to keep things straightforward.
    - Only querying one attribute.
    """
    def __init__(self) -> None:
        """ Constructor. Shut off access after 100 queries to not spill the beans. """
        self.allowable_number_of_queries = 100
        self.number_of_queries_executed = 0
        self.epsilon = 0.1

    @staticmethod
    def _calculate_sensitivity_single(query: str) -> Union[int, float, None]:
        """
        Calculate the sensitivity of a query.

        :param query: The original query. Parsed to determine sensitivity.
        :return: Sensitivity of query.
        """
        if query.__contains__("COUNT") or query.__contains__("count"):
            # Sensitivity is 1
            return 1

        elif query.__contains__("SUM") or query.__contains__("sum"):
            # Sensitivity is range(MAX, MIN) *MAX - MIN*
            # Parse query grabbed from URL
            query_as_list = query.split("+")
            temp_index = query_as_list.index("FROM")

            # Get attribute we want to find the MAX and MIN for
            attr = query_as_list[temp_index:][0]

            # Get max and min and return result of max - min
            max_result = DBManager.get_instance().execute(f"SELECT+MAX({attr})+FROM+PPP")
            min_result = DBManager.get_instance().execute(f"SELECT+MIN({attr})+FROM+PPP")

            return max_result[0][0] - min_result[0][0]

        elif query.__contains__("AVG") or query.__contains__("avg"):
            # Sensitivity is SUM / COUNT
            # Parse query grabbed from URL
            query_as_list = query.split("+")
            temp_index = query_as_list.index("FROM")

            # Get the attribute we want to find the SUM for
            attr = query_as_list[temp_index:][0]

            # Get sum and count and return result of sum / count
            sum_result = DBManager.get_instance().execute(f"SELECT+SUM({attr})+FROM+PPP")
            count_result = DBManager.get_instance().execute("SELECT+COUNT(*)+FROM+PPP")

            return sum_result[0][0] - count_result[0][0]

        else:
            return None

    def add_noise(self, query: str, result: Union[int, float]) -> Union[int, float, None]:
        """
        Add noise to a single value.

        :param query: Query executed on dataset.
        :param result: Original result.
        :return: Noisy result.
        """
        if self.number_of_queries_executed < self.allowable_number_of_queries:
            sensitivity = DFManager._calculate_sensitivity_single(query)
            noisy_result = self._apply_laplace_mechanism(result, sensitivity)
            self.number_of_queries_executed += 1
            return noisy_result

        else:
            return None

    def add_noise_parallel(self, query: str, result: List[Union[int, float]]) -> Union[List[Union[int, float]], None]:
        """
        Add noise to an array. Not using pd.Series because cannot be transferred over the network.

        :param query: Query executed on dataset.
        :param result: Original result.
        :return: Noisy array.
        """
        if self.number_of_queries_executed < self.allowable_number_of_queries:
            # TODO: Write out this method after verifying the single DP works
            self.number_of_queries_executed += 1

        else:
            return None

    def _apply_laplace_mechanism(self, value: Union[int, float], sensitivity: Union[int, float]) -> float:
        """
        Add noise to original value using the laplace mechanism

        :param value: Value to add noise to.
        :param sensitivity: Sensitivity of the query.
        :return: Noisy result.
        """
        return value + np.random.laplace(loc=0, scale=sensitivity/self.epsilon)


# URL route to use if we want to return the original query result
@app.route("/orig/<query>", methods=["GET"])
def orig(query):
    print(f"Received request for non-noisy answer for query {parse(query)}")
    result = DBManager.get_instance().execute(query)

    return jsonify({
        "query": parse(query),
        "result": result[0][0]
    })


# URL route to use if we want to return the differential private query result
@app.route("/diff/<query>", methods=["GET"])
def diff(query):
    print(f"Received request for noisy answer for query {parse(query)}")
    orig_result = DBManager.get_instance().execute(query)
    noisy_result = DF.add_noise(query, orig_result[0][0])

    return jsonify({
        "query": parse(query),
        "result": noisy_result
    })


if __name__ == "__main__":
    # Load SQLite database. Will take a couple of seconds. Make sure you have at last 1GB of RAM available
    # Dataset used: https://data.sba.gov/dataset/8aa276e2-6cab-4f86-aca4-a7dde42adf24/resource/cfd2e743-8809-49be-90b6-0e22f453be23/download/public_150k_plus_211121.csv
    print("Loading PPP Loan Dataset into SQLite...")
    DBManager.create_db("ppp_loan_data.csv")
    DF = DFManager()

    # Launch Flask web server on port 5000
    print("Starting Flask web server...")
    app.run(port=5000)
