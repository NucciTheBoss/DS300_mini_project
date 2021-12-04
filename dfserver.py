#!/usr/bin/env python3

import numpy as np
import pandas as pd
import sqlite3
from flask import Flask, jsonify
from typing import Any


# Initialize Flask web server
app = Flask(__name__)


class DBManager:
    """
    Singleton class to handle communication with SQLite database.
    """
    _class_instance = None
    _connection = None
    _cursor = None

    def __init__(self) -> None:
        """ Private constructor. """
        if DBManager._class_instance is not None:
            raise Exception("This class is a singleton.")

        else:
            DBManager._create_db("public_150k_plus_211121.csv")
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
    def _create_db(path_to_csv: str) -> None:
        """
        Create ephemeral SQL database in RAM. Database goes away after program ends.

        :param path_to_csv: Path to CSV file.
        """
        # Create DB in RAM and cursor to navigate database
        DBManager._connection = sqlite3.connect(":memory:")
        DBManager._cursor = DBManager._connection.cursor()

        # Read in csv file and add to the DB
        driving_data = pd.read_csv(path_to_csv)
        driving_data.to_sql("DRIVINGDATA", DBManager._connection, if_exists="replace")

    def execute(self, query: str) -> Any:
        """
        Execute query on SQLite DB.

        :param query: SQL query pulled from URL argument.
        :return: ResultSet or float/int/str value.
        """
        return DBManager._cursor.execute(self._parse(query))

    def _parse(self, query: str) -> str:
        """
        Parse URL query into correct SQL syntax.

        :param query: SQL query pulled from URL argument.
        :return: Parsed query converted to valid SQL syntax.
        """
        query_split = query.split("+")
        return " ".join(query_split)


class DF:
    """ Class to manage differential privacy mechanism and access to database. """
    def __init__(self) -> None:
        """ Constructor. """
        self.allowable_number_of_queries = 100
        self.number_of_queries_executed = 0


# URL route to use if we want to return the original query result
@app.route("/orig/<query>", methods=["GET"])
def orig(query):
    for row in DBManager.get_instance().execute(query):
        print(row)

    return jsonify({"success": 1})


# URL route to use if we want to return the differential private query result
@app.route("/diff/<query>", methods=["GET"])
def diff(query):
    print(query)
    return jsonify({"success": 1})


if __name__ == "__main__":
    # Launch Flask web server on port 5000
    app.run(port=5000)
