# (C) Copyright IBM Corp. 2020.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    'FlightService'
]

import os
import json
import sys
import threading
import requests
from typing import List, Optional, Iterable, TYPE_CHECKING
from .errors import (DataSourceTypeNotRecognized, DataStreamError, WrongLocationProperty,
                     WrongFileLocation, WrongDatabaseSchemaOrTable, MissingFileName, APIConnectionError)
from .utils import (prepare_cos_data_location, prepare_interaction_props_for_cos,
                    prepare_payload_for_excel, check_location, discover_input_data)

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from ibm_watson_machine_learning import APIClient

# the upper bound limitation for training data size
DATA_SIZE_LIMIT = 1073741824  # 1GB in Bytes


class FlightService:
    """FlightService object unify the work for training data reading from different types of data sources,
        including databases. It uses a Flight Service and `pyarrow` library to connect and transfer the data.

    Parameters
    ----------
    wml_client: APIClient, required
        WML Client

    entity: dictionary, required
        Overall description of the connection (fetched from connection API).
        Includes parameters that define connection method. Used by Flight Service payload.

    attachment: dictionary, required
        Information about Data Asset attachment downloaded from assets API.
        Should consists of e.g. path to the data in file storage or schema nad table names in database.

    n_batches: int, optional
        Defines for how many parts / batches data source should be split.
        Default is 1.

    data_location: dict, optional
        Data location information passed by user.

    params: dictionary, optional
        User defined KB parameters
    """

    def __init__(self,
                 wml_client: 'APIClient',
                 entity: dict,
                 attachment: dict,
                 n_batches: Optional[int] = 1,
                 data_location: dict = None,
                 params: dict = None) -> None:
        self.attachment = attachment
        self.wml_client = wml_client
        self._type = entity.get('datasource_type')
        self.name = entity.get('name')
        self.properties = entity.get('properties')
        self.n_batches = n_batches
        self.data_source_type = None

        self.data_location = data_location
        self.params = params

        self.lock_read = threading.Lock()
        self.stop_reading = False

        from pyarrow import flight
        self.flight_client = flight.FlightClient(
            location=f"grpc+tls://{os.environ.get('FLIGHT_SERVICE_LOCATION')}:{os.environ.get('FLIGHT_SERVICE_PORT')}",
            # TODO: change below params when in production
            disable_server_verification=True,
            override_hostname=os.environ.get('FLIGHT_SERVICE_LOCATION')
        )

    def authenticate(self) -> 'flight.ClientAuthHandler':
        """Creates an authenticator object for Flight Service."""
        from pyarrow import flight

        class TokenClientAuthHandler(flight.ClientAuthHandler):
            """Authenticator implementation from pyarrow flight."""

            def __init__(self, token):
                super().__init__()
                self.token = bytes('Bearer ' + token, 'utf-8')

            def authenticate(self, outgoing, incoming):
                outgoing.write(self.token)
                self.token = incoming.read()

            def get_token(self):
                return self.token

        return TokenClientAuthHandler(token=self.wml_client.wml_token)

    def get_endpoints(self) -> Iterable[List['flight.FlightEndpoint']]:
        """Listing all available Flight Service endpoints (one endpoint corresponds to one batch)"""
        from pyarrow import flight

        self.flight_client.authenticate(self.authenticate())
        try:

            for source_command in self._select_source_command():
                info = self.flight_client.get_flight_info(
                    flight.FlightDescriptor.for_command(source_command)
                )
                yield info.endpoints

        except flight.FlightInternalError as e:
            if 'CDICO2034E' in str(e):
                raise WrongLocationProperty(e)

            elif 'CDICO2015E' in str(e):
                raise WrongFileLocation(e)

            else:
                raise e

    def _get_data(self, endpoint: 'flight.FlightEndpoint') -> 'pd.DataFrame':
        """
        Read data from Flight Service (only one batch).

        Properties
        ----------
        endpoint: flight.FlightEndpoint, required

        Returns
        -------
        pd.DataFrame with batch data
        """
        from pyarrow import flight
        import pyarrow as pa

        try:
            reader = self.flight_client.do_get(endpoint.ticket)

        except flight.FlightUnavailableError as e:
            raise DataStreamError(reason=str(e))

        chunks = []

        # Flight Service could split one batch into several chunks to have better performance
        while True:
            try:
                chunk, metadata = reader.read_chunk()
                chunks.append(chunk)
            except StopIteration:
                break

        data = pa.Table.from_batches(chunks)
        return data.to_pandas()

    def _get_data_from_chunks(self, info) -> Iterable['pd.DataFrame']:
        """Iterator from all Flight Service endpoints."""
        for endpoint in info.endpoints:
            yield self._get_data(endpoint)

    def _prepare_command_for_db_select(self, select_statement: str) -> dict:
        """
        Helper method only for creating command for Flight Service with SELECT statement for databases.

        Parameters
        ----------
        select_statement: str, required
            SELECT statement for SQL query.

        Returns
        -------
        Dictionary with command for Flight Service.
        """
        command = {
            "datasource_type": {
                "entity": {
                    "name": self._type
                }
            },
            "connection_properties": self.properties,
            "interaction_properties": {
                "select_statement": select_statement
            },
            "num_partitions": 1
        }

        return command

    def _get_names(self) -> List[str]:
        """
        Prepare names which will be used to specify e.g. database schema, table or file bucket name.

        Returns
        -------
        List of strings.
        """
        try:
            names = self.attachment['connection_path'].split('/')

        except KeyError:
            if self.data_source_type == 'database':
                names = [
                    self.attachment['interaction_properties']['schema_name'],
                    self.attachment['interaction_properties']['table_name']
                ]

            elif self.data_source_type == 'file':
                names = [None]  # added to be compatible with previous split
                if 'bucket' in self.attachment['interaction_properties']:
                    names.append(self.attachment['interaction_properties']['bucket'])

                try:
                    names.append(self.attachment['interaction_properties']['file_name'])

                except KeyError as e:
                    raise MissingFileName(reason=e)

            elif self.data_source_type == 'generic':
                raise NotImplementedError(f"Data source type: {self.data_source_type} not implemented yet")

        return names

    def get_number_of_data_rows(self) -> int:
        """
        Calculates the number of data rows in connected database.

        Returns
        -------
        Integer with number of data rows.
        """
        from pyarrow import flight

        self.flight_client.authenticate(self.authenticate())
        names = self._get_names()

        command = self._prepare_command_for_db_select(f"SELECT count(*) FROM {names[-2]}.{names[-1]}")

        info = self.flight_client.get_flight_info(flight.FlightDescriptor.for_command(json.dumps(command)))
        data = self._get_data(info.endpoints[0])

        try:
            number_of_data_rows = data['count'].values[-1]

        except KeyError as e:
            number_of_data_rows = data.values[-1]  # MSSQL workaround

        return number_of_data_rows

    def get_row_size(self) -> int:
        """
        Calculates one data row size in Bytes.

        Returns
        -------
        Integer as a size of one data row in Bytes.
        """
        from pyarrow import flight
        import pyarrow as pa

        self.flight_client.authenticate(self.authenticate())
        names = self._get_names()

        try:
            command = self._prepare_command_for_db_select(f"SELECT * FROM {names[-2]}.{names[-1]} LIMIT 2")
            info = self.flight_client.get_flight_info(flight.FlightDescriptor.for_command(json.dumps(command)))

        except pa._flight.FlightInternalError as e:  # MSSQL specific SELECT statement
            command = self._prepare_command_for_db_select(f"SET ROWCOUNT 2; SELECT * FROM {names[-2]}.{names[-1]};")
            info = self.flight_client.get_flight_info(flight.FlightDescriptor.for_command(json.dumps(command)))

        data = self._get_data(info.endpoints[0])
        row_size = sys.getsizeof(data.iloc[:2]) - sys.getsizeof(data.iloc[:1])

        return row_size

    def get_data_size(self) -> int:
        """
        Calculates entire size of data stored in connected database.

        Returns
        -------
        Integer as a size of the entire data stored in DB.
        """
        data_size = self.get_row_size() * self.get_number_of_data_rows()

        return data_size

    def get_batch_limit(self) -> int:
        """
        Calculates upper limit for 'n_batches' that Flight Service can handle for specific data size.

        Returns
        -------
        Integer as a number of available batches.
        """
        max_rows = DATA_SIZE_LIMIT // self.get_row_size()
        batch_fraction = float(self.get_number_of_data_rows() / max_rows)

        if 0.1 <= batch_fraction <= 1:
            n_batches = 5  # try to find optimal number of batches under 1 GB data

        elif batch_fraction < 0.1:
            n_batches = 1  # data too small for batching

        else:
            n_batches = round(np.ceil(batch_fraction)) + 10  # split into more then 50 batches for data larger than 1 GB

        return int(n_batches)

    def _read_batch(self, batch_number: int, sequence_number: int,
                    endpoint: 'flight.FlightEndpoint', dfs: List['pd.DataFrame']) -> None:
        """This method should be used as a separate thread for downloading specific batch of data in parallel.

        Parameters
        ----------
        batch_number: int, required
            Specific number of the batch to download.

        sequence_number: int required
            Sequence number, it tells the number of file that we reads for file storage.

        endpoint: flight.FlightEndpoint, required
            Flight Service endpoint to connect.

        dfs: List['pd.DataFrame'], required
            List where we will be storing the downloaded batch. Shared across threads.
        """
        if not self.stop_reading:
            df = self._get_data(endpoint)
            row_size = sys.getsizeof(df.iloc[:2]) - sys.getsizeof(df.iloc[:1])

            with self.lock_read:
                if not self.stop_reading:
                    # note: what to do if we have batch too large (over 1 GB limit)
                    if sys.getsizeof(df) > DATA_SIZE_LIMIT:  # this is GB in Bytes
                        row_limit = round(DATA_SIZE_LIMIT / row_size)
                        dfs.append(df.iloc[:row_limit])
                        self.stop_reading = True
                    # --- end note
                    else:
                        total_size = sum([row_size * len(data) for data in dfs])

                        # note: what to do when we have total size nearly under the limit
                        if total_size <= DATA_SIZE_LIMIT:
                            upper_row_limit = (DATA_SIZE_LIMIT - total_size) // row_size
                            df = df.iloc[:upper_row_limit]
                            dfs.append(df)
                        # --- end note

                        else:
                            self.stop_reading = True

            print(f"Downloaded batch number: {batch_number}, sequence: {sequence_number}")
            print(f"Batch shape: {df.shape}")
            print(f"Estimated batch size: {sys.getsizeof(df)} Bytes")
            print(f'Thread {batch_number}, sequence: {sequence_number} completed reading the batch.')

    def read(self) -> 'pd.DataFrame':
        """Fetch the data from Flight Service. Fetching is done in batches.
            There is an upper top limit of data size to be fetched configured to 1 GB.

        Returns
        -------
        Pandas DataFrame with fetched data.
        """
        dfs = []
        sequences = []

        # Note: endpoints are created by Flight Service based on number of partitions configured
        # one endpoint serves one batch of the data
        for n, endpoints in enumerate(self.get_endpoints()):
            threads = []
            sequences.append(threads)

            for i, endpoint in enumerate(endpoints):
                reading_thread = threading.Thread(target=self._read_batch, args=(i, n, endpoint, dfs))
                threads.append(reading_thread)
                print(f"Starting batch reading thread: {i}, sequence: {n}...")
                reading_thread.start()

        for n, sequence in enumerate(sequences):
            for i, thread in enumerate(sequence):
                print(f"Joining batch reading thread {i}, sequence: {n}...")
                thread.join()

        dfs = pd.concat(dfs)

        # Note: be sure that we do not cross upper data size limit
        estimated_data_size = sys.getsizeof(dfs)
        estimated_row_size = estimated_data_size // len(dfs)
        size_over_limit = estimated_data_size - DATA_SIZE_LIMIT

        if size_over_limit > 0:
            upper_limit = len(dfs) - (size_over_limit // estimated_row_size) * 2
            dfs = dfs.iloc[:upper_limit]
        # --- end note

        print(f"TOTAL DOWNLOADED DATA SIZE: {sys.getsizeof(dfs)} Bytes")

        return dfs

    def _list_data_source_types(self) -> List[dict]:
        """Listing all available data source types and IDs as WML client
            does not return an object but rather displays a table.
        """
        # note: list all available connection data types
        if self.wml_client.ICP:
            response = requests.get(self.wml_client.connections._href_definitions.get_connection_data_types_href(),
                                    params=self.wml_client._params(),
                                    headers=self.wml_client._get_headers(),
                                    verify=False)

        else:
            response = requests.get(self.wml_client.connections._href_definitions.get_connection_data_types_href(),
                                    params=self.wml_client._params(),
                                    headers=self.wml_client._get_headers())

        if response.status_code == 200:
            return response.json()['resources']

        else:
            raise APIConnectionError('datasource_types endpoint',
                                     reason=f'ERROR: {response.status_code}, {response.reason}')
        # --- end note

    def _select_source_command(self) -> List[str]:
        """Based on a data source type, select appropriate commands for flight service configuration."""
        from pyarrow import flight

        data = self._list_data_source_types()

        # note: retrieve data source type name
        for data_source in data:
            if data_source['metadata']['asset_id'] == self.attachment['datasource_type']:
                self.data_source_type = data_source['entity']['type']
                break
        # --- end note

        # TODO: change this implementation to be more general
        try:
            del self.properties['username_password_security']
            del self.properties['username_password_encryption']

        except:
            pass

        commands = []

        if self.data_source_type == 'database':
            check_location(self.data_location, self.data_source_type)
            try:
                self.n_batches = self.get_batch_limit()

            except flight.FlightInternalError as e:
                if 'CDICO2005E' in str(e):
                    raise WrongDatabaseSchemaOrTable(e)

                else:
                    raise e

            names = self._get_names()

            command = {
                "datasource_type": {
                    "entity": {
                        "name": self._type
                    }
                },
                "connection_properties": self.properties,
                "interaction_properties": {
                    "schema_name": names[-2],
                    "table_name": names[-1]
                },
                "num_partitions": self.n_batches
            }

            commands.append(json.dumps(command))

        elif self.data_source_type == 'file':
            check_location(self.data_location, self.data_source_type)
            cos_data_location = prepare_cos_data_location(data_location=self.data_location,
                                                          wml_client=self.wml_client)
            input_key_files = discover_input_data(cos_data_location)
            for input_key_file in input_key_files:
                cos_interaction_properties = prepare_interaction_props_for_cos(params=self.params,
                                                                               input_key_file=input_key_file)

                cos_interaction_properties, input_key_file = prepare_payload_for_excel(
                    cos_data_location, cos_interaction_properties, self.params, input_key_file)

                names = self._get_names()

                command = {
                    "datasource_type": {
                        "entity": {
                            "name": self._type
                        }
                    },
                    "connection_properties": self.properties,
                    "interaction_properties": {
                        "infer_schema": "true",
                        "file_name": input_key_file,  # A path with bucket is fine
                        "bucket": self.properties["bucket"] if "bucket" in self.properties else names[1]
                    },
                    "num_partitions": self.n_batches
                }

                command["interaction_properties"].update(cos_interaction_properties)

                commands.append(json.dumps(command))

        elif self.data_source_type == 'generic':
            raise NotImplementedError(f"Data source type: {self.data_source_type} not implemented yet")

        else:
            raise DataSourceTypeNotRecognized(
                self.data_source_type,
                reason=f"Flight Service cannot operate on {self.data_source_type} data source type!")

        return commands
