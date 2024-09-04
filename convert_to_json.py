import pandas as pd
import json

csv_file = 'train.csv' 
chat_df = pd.read_csv(csv_file,sep=',')

json_output = r"D:\hugging_face_ft\chat_data.json"
output = chat_df.to_json(json_output,force_ascii=False,indent=1,orient='records')
