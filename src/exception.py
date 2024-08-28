import sys
from src.logger import logging  # Ensure that the logger module is correctly imported

def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information.
    
    :param error: The error message.
    :param error_detail: The sys module to extract detailed traceback information.
    :return: Formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    """
    Custom Exception class to capture and log detailed error messages.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

# Example usage of CustomException
try:
    # Some code that may raise an exception
    raise ValueError("This is a sample error")
except ValueError as e:
    # Log the error or handle it as needed
    logging.error(CustomException(str(e), sys))
