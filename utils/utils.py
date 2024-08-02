import os


class Utils:

    def __init__(self):
        pass

    def get_data_path(self, folder_path):
        file_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        return file_paths
