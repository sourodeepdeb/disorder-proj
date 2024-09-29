import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import ast
import time
from tqdm import tqdm
from openai import OpenAI
from google.colab import drive
import OpenAI

drive.mount('/content/gdrive')
dir_path = "/content/gdrive/My Drive/disorder_proj/"

suicide_files = ["SuicideWatch_posts.txt"]
stress_files = ["Stress_posts.txt"]
normal_files = ["normal_posts.txt"]
ocd_files = ["OCD_posts.txt"]
anxiety_files = ["Anxiety_posts.txt"]

def read_all_data(file_names, target):
    all_lines = []
    for fname in file_names:
        file_path = dir_path + fname
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith(("Title", "Post")):
                    all_lines.append(line.strip())

    df = pd.DataFrame()
    df["text"] = all_lines
    df["target"] = target

    return df

df_suicide = read_all_data(suicide_files, 0)
df_stress = read_all_data(stress_files, 1)
df_normal = read_all_data(normal_files, 2)  
df_ocd = read_all_data(ocd_files, 3)
df_anxiety = read_all_data(anxiety_files, 4)

print("Size of Suicide DataFrame:", df_suicide.size)
print("Size of Stress DataFrame:", df_stress.size)
print("Size of Normal DataFrame:", df_normal.size)  
print("Size of OCD DataFrame:", df_ocd.size)
print("Size of Anxiety DataFrame:", df_anxiety.size)

df_data = pd.concat([df_suicide, df_stress, df_ptsd, df_ocd, df_anxiety], axis=0, ignore_index=True)

OPEN_API_KEY = "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"
client = OpenAI(api_key=OPEN_API_KEY)

import time
from tqdm import tqdm

OPEN_API_KEY = "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"
client = OpenAI(api_key=OPEN_API_KEY)

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

print("Generating embeddings...")
start_time = time.time()

tqdm.pandas()
df_data['ada_embedding'] = df_data.text.progress_apply(
    lambda x: get_embedding(x, model='text-embedding-3-small')
)

end_time = time.time()
print(f"Time taken to generate embeddings: {end_time - start_time:.2f} seconds")

df_data.to_csv(dir_path + "training_embedding.csv", index=False)
print("Saved to:", dir_path + "training_embedding.csv")

OPEN_API_KEY = "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"
client = OpenAI(api_key=OPEN_API_KEY)

dir_path = "/content/gdrive/My Drive/disorder_proj/"

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def read_normal_data(file_name, target):
    with open(dir_path + file_name, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip() and not line.startswith(("Title", "Post"))]

    df = pd.DataFrame({"text": all_lines, "target": target})
    return df

df_existing = pd.read_csv(dir_path + "training_embedding.csv")

df_normal = read_normal_data("normal_posts.txt", 2) 

print("Generating embeddings for normal posts...")
start_time = time.time()

tqdm.pandas()
df_normal['ada_embedding'] = df_normal.text.progress_apply(
    lambda x: get_embedding(x, model='text-embedding-3-small')
)

end_time = time.time()
print(f"Time taken to generate embeddings: {end_time - start_time:.2f} seconds")

df_combined = pd.concat([df_existing, df_normal], ignore_index=True)

df_combined.to_csv(dir_path + "training_embedding_updated.csv", index=False)
print("Saved to:", dir_path + "training_embedding_updated.csv")
