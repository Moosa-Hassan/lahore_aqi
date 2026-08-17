from dotenv import load_dotenv
import os

load_dotenv()
print('HOPSWORKS_API_KEY:', os.environ.get('HOPSWORKS_API_KEY')[:40] if os.environ.get('HOPSWORKS_API_KEY') else 'NOT FOUND')