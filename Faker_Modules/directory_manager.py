import os

def create_directory(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created successfully.")
    else:
        print(f"Directory {directory} already exists.")
