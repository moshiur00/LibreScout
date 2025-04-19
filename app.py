from dotenv import load_dotenv
import os
import openai
from interface import load_interface

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

load_interface()