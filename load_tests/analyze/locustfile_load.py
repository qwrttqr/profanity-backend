import pandas as pd
from locust import HttpUser, task, between

df = pd.read_csv('../../test.csv')

class AnalyzeTester(HttpUser):
    # Simulate user waiting between 1 and 3 seconds between tasks
    wait_time = between(1, 3)
    @task
    def analyze_text(self):
        for index, row in df.iterrows():
            text_value = row['text']
            self.client.get(f'/analyze/?text={text_value}')