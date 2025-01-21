import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information including the script name, line number, and error message.

    Args:
        error (Exception): The exception instance.
        error_detail (sys): The sys module to access exception details.

    Returns:
        str: Formatted error message with detailed information.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in script: [{file_name}] "
        f"at line number: [{exc_tb.tb_lineno}] "
        f"with message: [{str(error)}]"
    )
    return error_message

class CustomException(Exception):
    """
    Custom exception class for handling and logging detailed error messages.

    Args:
        error_message (str): The error message to be logged.
        error_detail (sys): The sys module to access exception details.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message

# # Testing the CustomException
# if __name__ == "__main__":
#     try:
#         a = 1 / 0  # Example of an error
#     except Exception as e:
#         raise CustomException("Division by zero error", sys)
