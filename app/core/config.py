from pydantic_settings import BaseSettings
import os
from pydantic import Field 
from typing import List


# env경로 절대경로. env파일은 app폴더밖에 둬주세요.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Downloads경로 절대경로로 설정 ~/Downloads
DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Downloads")

class Settings(BaseSettings):
    PROJECT_NAME: str = "법률 챗봇"

    # 모델설정
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434") #모델 url
    # 역할에 따라 모델 분리
    # str은 헌팅식 python 문법입니다. typescript처럼 타입을 미리 정의를 해주는방식
    SUMMARIZE_MODEL: str = Field(default="exaone3.5:2.4b")
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3")    # 임베딩용
    RAG_MODEL: str = "law-exaone-7.8b-q6"        # RAG 응답용

    #임베딩모델 설정
    SPARSE_THRESHOLD: float = Field(default=...)
    EMBEDDING_DEVICE: str= Field(default=...) 
    EMBEDDING_BATCH: int= Field(default=...)

    # 허용 주소값
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173", "https://www.legalchat.shop"]

    # pincone
    # `...`(Ellipsis)은 "기본값은 비운다는 의미지만 env에서 시스템환경변수 같은 값을 최우선으로 가져오는 의미입니다.
    PINECONE_API_KEY: str = Field(default=...)
    PINECONE_INDEX_NAME: str = Field(default=...)
    PINECONE_CLOUD: str = Field(default=...)
    PINECONE_REGION: str = Field(default=...)

    # Auth 세팅
    SECRET_KEY: str = Field(default=...)
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60;           # 1시간
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 1; # 1일
    
    # Postgres 세팅
    POSTGRES_SERVER: str = Field(default=...)

    # Upload Directory 세팅
    UPLOAD_DIR: str = DOWNLOAD_DIR+"/uploads"

    # Oauth 세팅
    KAKAO_ADMIN_KEY: str = Field(default=...)
    KAKAO_REST_API_KEY: str = Field(default=...)
    KAKAO_REDIRECT_URI: str = Field(default=...)
    KAKAO_CLIENT_SECRET: str = Field(default=...)

    # Admin 계정 세팅
    ADMIN_EMAIL: str = Field(default=...)
    ADMIN_PASSWORD: str = Field(default=...)
    ADMIN_NAME: str = Field(default=...)

    # Multi-Centroid 설정
    CENTROID_MODEL_PATH: str = Field(
        default=os.path.join(
            BASE_DIR,
            "app",
            "centroid_dataset",
            "multi_centroids.npz",
        )
    )
    CENTROID_MIN_SCORE: float = Field(default=0.5271)
    CENTROID_MIN_MARGIN: float = Field(default=0.05)

    # Cross-Encoder 설정
    RERANKER_MODEL: str = Field(
        default="BAAI/bge-reranker-v2-m3"
    )
    RERANKER_DEVICE: str = Field(default="cpu")


    class Config:
        env_file = BASE_DIR+"/.env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# 지울거
SECRET_KEY: str = Field(default=...)
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 60