#!/usr/bin/env python
# coding: utf-8

# # 1. 라이브러리 선언

# In[1]:


# 서버 관리용 fastapi 의존 라이브러리
import uvicorn


# In[2]:


# 비동기 처리가 가능한 파이썬 웹 서버 라이브러리
from fastapi import FastAPI


# In[3]:


# 데이터 바이너리 저장용 라이브러리 (특징: 얘는 데이터 타입을 그대로 보존한다)
import pickle


# In[4]:


# 데이터 행과열을 처리하는 라이브러리
import pandas as pd
# 데이터 수를 관리하는 라이브러리
import numpy as np


# In[5]:


# 인터페이스 데이터 관리를 위한 라이브러리
from pydantic import BaseModel


# # 2. 모델 불러오기

# In[6]:


# scikit-learn 버전 맞추기
with open("kopo_mlcore.dump", "rb") as fr:
    loadedModel = pickle.load(fr)


# In[7]:


# 웹서버 기동
app = FastAPI(title="ML API")


# In[8]:


# !pip list


# In[9]:


from sklearn.tree import plot_tree


# # 3.인터페이스 정의

# In[10]:


# HCLUS, PROMOTION, HOLIDAY YOR (Y:1, N:0) PROMOTION (Y:1, N:0)
# features = ["LE_HCLUS", "PRO_PERCENT", "LE_HOLIDAY", "LE_PROMOTION"]
# label = ["QTY"]
class InDataset(BaseModel):
    Hclus: int
    Propercent: float
    Holiday: int
    Promotion: int


# # 4. 라우터 정의

# In[11]:


@app.get("/")
async def root():
    return {"message":"hello server is running"}


# In[12]:


@app.post("/predict", status_code=200)
async def predict_ml(x: InDataset):
    # 1-4 4는 소규모 홀리데이 1은 대규모 홀리데이 (블랙프라이데이)
    # inHclus = 0
    # inPropercent = 0.0
    # inHoliday = 0
    # inPromotion = 0
    print(x)
    testDf = pd.DataFrame( [[x.Hclus,x.Propercent,x.Holiday,x.Promotion ]])
    predictValue = int( loadedModel.predict( testDf )[0] )
    interfaceResult = {"result": predictValue }
    return interfaceResult

# # 서버구동

# In[13]:


import uvicorn


# In[ ]:


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9999, log_level="debug",
                proxy_headers=True, reload=True)


# In[ ]:


# features = ["LE_HCLUS", "PRO_PERCENT", "LE_HOLIDAY", "LE_PROMOTION"]


# In[ ]:


# 1-4 4는 소규모 홀리데이 1은 대규모 홀리데이 (블랙프라이데이)
# inHclus = 1
# inPropercent = 0.5
# inHoliday = 1
# inPromotion = 1


# In[ ]:


# 1-4 4는 소규모 홀리데이 1은 대규모 홀리데이 (블랙프라이데이)
inHclus = 0
inPropercent = 0.0
inHoliday = 0
inPromotion = 0


# In[ ]:


pd.DataFrame( [[inHclus,inPropercent,inHoliday,inPromotion ]])


# In[ ]:


testDf = pd.DataFrame( [[inHclus,inPropercent,inHoliday,inPromotion ]])


# In[ ]:


loadedModel.predict( testDf )


# In[ ]:


predictValue = loadedModel.predict( testDf )
predictValue


# In[ ]:


predictValue = loadedModel.predict( testDf )[0]
predictValue


# In[ ]:


predictValue = int( loadedModel.predict( testDf )[0] )
interfaceResult = {"result": predictValue }

# loadedModel
# In[ ]:


# import matplotlib.pyplot as plt

# plt.figure(figsize = (15,8))
# plot_tree(decision_tree = loadedModel)


# In[ ]:


# 프레임워크 반제품.... 개발을 편리하게 하도록
# 일반적인 로그처리 기본 기능을 탑재한 환경!!!
# java spring~!!!
# python django flask fastapi
# node express....

