import warnings
warnings.filterwarnings('ignore')
import os

#! this line is needed for code to execute properly
path = os.path.expanduser("")

from pydantic.v1 import BaseModel
from crewai_tools import BaseTool
import os

class FolderInput(BaseModel):
    folder_path: str

class FileReaderTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="file_reader",
            description="Reads all text files from a folder and returns their contents.",
            args_schema=FolderInput
        )

    def _run(self, folder_path: str):
        if not os.path.isdir(folder_path):
            return "Provided path is not a directory."
        
        file_contents = {}
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_contents[filename] = f.read()
                except Exception as ex:
                    file_contents[filename] = f"Error reading file: {str(ex)}"

        return file_contents

class FileCleaner:
    def __init__(self):
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

        from crewai import Agent, Task, Crew

        self.file_reader = Agent(
            role = "Computer File Reader",
            goal = "Read through every text file within {target_folder} and generate a detailed summary of the contents of each text file.",
            backstory = "You are an ace at reading through large text files and generating precise, detailed summaries of them."
                        "You work at a very large firm and have decades of experience doing this.",
            verbose = True,
            tools = [FileReaderTool()]
        )

        self.standard_analyzer = Agent(
            role = "File Standard Analyzer",
            goal = "Read through all text files within {standard_folder} and generate a detailed list of criteria that an ideal folder has."
                   "The ideal folder contains only good text files. The contents of the file text files are to be analyzed to attribute them "
                   "based on criteria like date, content type, etc.",
            verbose = True,
            tools = [FileReaderTool()],
            backstory = "You are an expert at reading and understanding files and giving them tags based on their attributes as you have been "
                        "in this position for forty years. You look out for things like the dates of files, whether they contain temporary or "
                        "long-term relavent information, etc. to develop a profile of how a company likes to keep their files.",
        )

        self.cleanup_recommender = Agent(
            role = "File Cleanup Recommender",
            goal = "Create a list of file names that can be deleted as they do not contain important information. This list must include rationale.",
            verbose = True,
            tools = [FileReaderTool()],
            backstory = "You have a keen and experienced eye to compare the ideal profile of what a company wants their folders to be like with "
                        "a folder selected for cleanup. Through your exceptional analysis, you are able to select which files in {target_folder} "
                        "a company should delete. You don't delete essential information, but you are also not conservative, and have unmatched judgement.",
        )

        self.target_read = Task(
            description = """
            Read through all text files in {target_folder} and summarize every file in there.
            """,
            expected_output = """
            A detailed summary of the contexts of every file.
            """,
            agent = self.file_reader,
        )

        self.standard_read = Task(
            description = "Read through all text files in {standard_folder} to develop a profile for what a good folder looks like.",
            expected_output = "A detailed break down of what a good folder of text files contains."
                              "This should include specific attributes of a good text file that can be used for comparison.",
            agent = self.standard_analyzer, 
        )

        self.recommend = Task(
            description = "Using the created profile of what a good folder looks like and the summary of what each file contains, "
                          "create a list of files that should be removed. Provide a few sentences of rationale for each file.",
            expected_output = "A list of files that should be removed with a few sentences of rationale for each file."
                              "These files are also categorized based on whether they are useless, may be needed for specific uses, "
                              "and/or need further human review.",
            agent = self.cleanup_recommender,
        )

        self.file_cleaner_crew = Crew(
            agents = [
                self.file_reader,
                self.standard_analyzer,
                self.cleanup_recommender,
            ],
            tasks = [
                self.target_read,
                self.standard_read,
                self.recommend,
            ],
            verbose = True,
        )
    
    def kickoff(self):
        self.inputs = {
            'target_folder' : os.path.expanduser("~/ITCS/Senior Summit/FileCleaner/TestFolder/target_folder"),
            'standard_folder' : os.path.expanduser("~/ITCS/Senior Summit/FileCleaner/TestFolder/standard_folder")
        }

        return self.file_cleaner_crew.kickoff(inputs = self.inputs)
    
if __name__ == '__main__':
    cleaner = FileCleaner()
    result = cleaner.kickoff()
    print(result)

