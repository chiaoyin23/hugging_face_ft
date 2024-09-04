### 1. 終端機先登入 hugging_face ###
```
huggingface-cli login
```
### 2. 輸入 hugging_face access token ###
 huggingface ：[https://huggingface.co/](https://huggingface.co)
### 3. 準備 csv 檔案轉換成 json ###
```
python convert_to_json.py
```
執行成功後生成 chat_data.json

### 4. 安裝需要的東西 (應該不多) ###
```
pip install transformers,  datasets
```
### 5. 跑微調，完成後 base_model 存在本地` ./saved_taide_base_model `； finetune_model 存在`./finetuned-model` ###
```
python finetune.py
```

### 6.微調後可以用用看模型效果如何 ### 
```
python inference.py
```
