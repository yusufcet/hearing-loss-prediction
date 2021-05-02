__all__ = [
    'DataSourceTypeNotRecognized',
    'LabelNotFound',
    'APIConnectionError',
    'DataStreamError',
    'WrongDatabaseSchemaOrTable',
    'WrongFileLocation',
    'WrongLocationProperty',
    'UnsupportedConnection',
    'UnsupportedOutputConnection',
    'MissingFileName',
    'FileUploadFailed'
]

from ibm_watson_machine_learning.wml_client_error import WMLClientError


class DataSourceTypeNotRecognized(WMLClientError, NotImplementedError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"Data source type: {value_name} not recognized!", reason)


class LabelNotFound(WMLClientError, KeyError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"Cannot find label: {value_name}", reason)


class APIConnectionError(WMLClientError, ConnectionError):
    def __init__(self, value_name, reason=None):
        WMLClientError.__init__(self, f"Cannot connect to: {value_name}", reason)


class DataStreamError(WMLClientError, ConnectionError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, "Cannot fetch data via Flight Service. Try again.", reason)


class WrongLocationProperty(WMLClientError, ConnectionError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, "Cannot fetch data via Flight Service. Try again.", reason)


class WrongFileLocation(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, "Cannot fetch data via Flight Service. Try again.", reason)


class WrongDatabaseSchemaOrTable(WMLClientError, ValueError):
    def __init__(self, value_name=None, reason=None):
        WMLClientError.__init__(self, "Cannot fetch data via Flight Service. Try again.", reason)


class UnsupportedOutputConnection(WMLClientError, ValueError):
    def __init__(self, connection_id=None, reason=None):
        WMLClientError.__init__(self, f"Connection with ID: {connection_id} is not supported as an Output connection",
                                reason)


class UnsupportedConnection(WMLClientError, ValueError):
    def __init__(self, conn_type=None, reason=None):
        WMLClientError.__init__(self, f"Connection type: {conn_type} is not supported.",
                                reason)


class MissingFileName(WMLClientError, KeyError):
    def __init__(self, reason=None):
        WMLClientError.__init__(self, f"Connection location requires a 'file_name' to be specified.",
                                reason)


class FileUploadFailed(WMLClientError, ConnectionError):
    def __init__(self, reason=None):
        WMLClientError.__init__(self, f"Failed to upload file.",
                                reason)
