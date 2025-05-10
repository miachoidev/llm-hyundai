# 🚄 현대로템 공고사양 자동 추출 시스템

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-0B968F?style=for-the-badge&logo=OpenAI&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-7A1FA2?style=for-the-badge&logo=Langchain&logoColor=white)

## 📋 프로젝트 개요

본 프로젝트는 현대로템의 열차 공고사양서에서 필수 기술 사양을 자동으로 추출하고 분석하는 RAG(Retrieval Augmented Generation) 기반 시스템입니다. 대규모 사양서에서 중요 정보를 빠르게 찾고 분석하는 작업을 자동화하여 업무 효율을 크게 향상시킵니다.

<p align="center">
  <img src="https://raw.githubusercontent.com/miachoidev/llm-hyundai/main/docs/demo_result.png" alt="main" width="800"/>
  <br>
  <em>시스템 데모 화면</em>
 
</p>

## 📱 데모
 [ 🌐 프로젝트 배포 링크](https://llm-hyundai-gx3meqkrkwmfkcr2692yor.streamlit.app/)


## ✨ 주요 기능

- **문서 분석**: DOCX 형식의 공고사양서를 자동으로 분석
- **사양 추출**: 문서 내 핵심 기술 사양 자동 추출
- **정확도 평가**: 추출된 사양의 정확도를 자동으로 검증
- **검색 성능 측정**: Recall@K 방식으로 검색 성능 측정

## 🔧 기술 스택

- **프론트엔드**: Streamlit
- **백엔드**: Python
- **자연어 처리**: LLM(gpt-4o-mini)
- **문서 처리**: python-docx, docx2txt
- **벡터 검색**: Chroma DB, BM25 검색
- **임베딩**: OpenAI 임베딩

## 🏗️ 시스템 아키텍처

이 시스템은 RAG(Retrieval Augmented Generation) 아키텍처를 기반으로 구축되었습니다. 아래 다이어그램은 전체 시스템의 작동 방식을 보여줍니다:

<p align="center">
  <img src="https://raw.githubusercontent.com/miachoidev/llm-hyundai/main/docs/architecture.png" alt="시스템 아키텍처" width="800"/>
  <br>
  <em>시스템 아키텍처 다이어그램</em>
</p>

### 워크플로우

1. **문서 처리**: 공고사양서(DOCX)를 업로드하면 시스템이 자동으로 구조를 분석하고 청킹합니다.
2. **벡터 저장소 생성**: 분석된 문서 청크는 OpenAI 임베딩을 통해 벡터화되어 ChromaDB에 저장됩니다.
3. **하이브리드 검색**: 사용자 쿼리에 대해 BM25와 벡터 검색을 결합한 하이브리드 검색을 수행합니다.
4. **LLM 추론**: 검색된 관련 컨텍스트를 기반으로 LLM(gpt-4o-mini)이 최종 답변을 생성합니다.
5. **성능 평가**: 시스템은 자동으로 생성된 답변과 정답을 비교하여 성능을 평가합니다.

## 🚀 시작하기

### 필수 조건

- Python 3.8 이상
- OpenAI API 키

### 설치 방법

```bash
# 저장소 클론
git clone https://github.com/miachoidev/llm-hyundai.git
cd llm-hyundai

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가하세요:

```
OPENAI_API_KEY=your_openai_api_key
```

### 실행 방법

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하여 앱을 이용할 수 있습니다.

## 📊 사용 방법

1. 사이드바에서 사양서 파일(DOCX)을 업로드합니다
2. LLM 모델을 선택합니다 (기본: gpt-4o-mini)
3. '사양 추출 시작' 버튼을 클릭합니다
4. 결과를 확인하고 필요시 CSV로 다운로드합니다




</details>

## 🔍 주요 구성 요소

- **app.py**: 메인 Streamlit 애플리케이션
- **chunker.py**: 문서 처리 및 청킹 로직
- **data/**: 참조 데이터 및 정답 데이터셋

## 📈 성능 및 벤치마크

시스템은 다음 지표로 성능을 평가합니다:
- **LLM 추론 성능**: 정답 기준 F1 / EM(Exact Match)
- **RAG 검색 성능**: Recall@K 측정 방식

<p align="center">
  <img src="https://raw.githubusercontent.com/miachoidev/llm-hyundai/main/docs/performance.png" alt="성능 그래프" width="600"/>
  <br>
  <em>성능 평가 결과</em>
</p>



## 🙋‍♂️ Contact Me

📧 **Email**  
mia.choi.dev@gmail.com

💼 **LinkedIn**  
[Linked-in](www.linkedin.com/in/mia-b43a04340)

📓 **Portfolio (Notion)**  
[Mia's Portfolio](https://vivivic777.notion.site/AI-MIA-1e5040331611807b8dc8c7447e3ae6e3)


