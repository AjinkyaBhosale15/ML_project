import sys
import logging

# Logging configuration (if not already done)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract detailed error information (filename, line number, error message)
def error_message_detail(error: Exception, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # Extract the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the filename where the error occurred
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

# Custom exception class for detailed error logging and management
class CustomException(Exception):
    def __init__(self, error: Exception, error_details: sys):
        """
        Custom exception that captures and logs detailed error information.
        """
        super().__init__(str(error))  # Pass the error message to the base Exception class
        self.error_message = error_message_detail(error, error_details)  # Generate the detailed error message

    def __str__(self):
        return self.error_message  # Return the detailed error message when printed

# Example usage
try:
    # Simulate some operation that may throw an exception
    pass
except Exception as e:
    logging.error("An error occurred!")  # Log the error
    raise CustomException(e, sys)  # Raise the custom exception with detailed information
