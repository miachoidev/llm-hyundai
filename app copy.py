import streamlit as st
import pandas as pd
import os
import tempfile
import re
from difflib import SequenceMatcher
from langchain_core.documents import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from chunker import convert_docx_to_chunks
from langchain_community.vectorstores.utils import filter_complex_metadata

# 페이지 설정
st.set_page_config(page_title="열차 사양서 분석기", page_icon="🚄", layout="wide")

# 제목 및 설명
st.title("🚄 현대로템 - 열차 공고사양 자동 분석(PoC)")
st.markdown(
    """📄 본 시스템은 PoC(Proof of Concept)용으로, 공고 사양서 기반 사양 추출 및 평가 자동화를 시연합니다.  
🧪 사전에 정답이 포함된 데이터셋으로 LLM 추론 및 검색 성능을 평가합니다.  

🔧 사용 방법:  
1️⃣ 열차 제작 공고 사양서를 업로드 후 모델을 선택하고 '사양 추출 시작' 버튼을 누르세요.  
2️⃣ RAG (Retrieval-Augmented Generation) 기법을 이용하여 문서에서 스펙 정보를 자동 추출합니다.  
3️⃣ 분석 결과는 CSV 파일로 다운로드 가능합니다.  

📊 모델 평가: 본 PoC는 사양서 기반 정보 추출 정확도와 검색 재현율이 핵심  
LLM 추론 성능 평가 : 정답 기준 F1 / EM(Exact Match)  
RAG 검색 성능 측정 : Recall@K 방식으로 측정  
"""
)


# 정답 평가 함수 정의
def extract_numbers(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"\d+\.?\d*", text)


def is_answer_correct(pred, answer):
    if not isinstance(pred, str) or not isinstance(answer, str):
        return False

    pred_numbers = extract_numbers(pred)
    answer_numbers = extract_numbers(answer)

    # 숫자 비교 우선
    if pred_numbers and answer_numbers:
        return pred_numbers == answer_numbers

    # 숫자가 없거나 비교 불가능하면 문자열 유사도 비교 (서술형)
    similarity = SequenceMatcher(None, pred.strip(), answer.strip()).ratio()
    return similarity >= 0.9  # 임계값은 상황에 따라 조정 가능


def is_correct_chunk(answer_chunk, reference_chunks):
    # 문자열 비교가 아닌 Document 객체 리스트에서 정답 문서 내용 확인
    if isinstance(reference_chunks, str) and "Document" in reference_chunks:
        try:
            # 참조 문서가 문자열로 저장된 경우, page_content 값들을 추출
            for chunk in reference_chunks.split("Document("):
                if answer_chunk in chunk and "page_content" in chunk:
                    # page_content 부분에서 정답 문서가 포함되어 있는지 확인
                    content_part = (
                        chunk.split("page_content='")[1].split("')")[0]
                        if "page_content='" in chunk
                        else ""
                    )
                    if content_part and answer_chunk in content_part:
                        return True
            return False
        except Exception:
            # 파싱 에러 발생 시 기존 방식으로 비교
            return answer_chunk == reference_chunks
    else:
        # 기존 방식 유지 (문자열 직접 비교)
        return answer_chunk == reference_chunks


# CSV 파일에서 테이블 데이터 로드
# @st.cache_data - 캐시 제거 (개발 중 변경사항 즉시 확인 위함)
def load_table_data():
    try:
        # data 폴더에서 CSV 파일 로드
        csv_path = os.path.join("data", "hd_table 3.csv")
        df = pd.read_csv(csv_path, encoding="utf-8")
        return df
    except Exception as e:
        st.error(f"CSV 파일 로드 중 오류 발생: {e}")
        # 오류 발생 시 기본 데이터프레임 반환
        return pd.DataFrame(
            {
                "레이블1": ["오류", "CSV 파일을 찾을 수 없습니다"],
                "레이블2": ["", ""],
                "레이블3": ["", ""],
                "레이블4": ["", ""],
                "표준단위": ["", ""],
                "정답": ["", ""],  # 정답 컬럼 추가
            }
        )


def generate_queries(row):
    """여러 방식으로 쿼리 생성"""
    queries = []

    # 1) 레벨2,3,4,표준단위 조합
    # nan이나 빈 문자열 제외하고 문자열 생성
    level2 = (
        row.get("레벨2", "")
        if row.get("레벨2", "") and pd.notna(row.get("레벨2", ""))
        else ""
    )
    level3 = (
        row.get("레벨3", "")
        if row.get("레벨3", "") and pd.notna(row.get("레벨3", ""))
        else ""
    )
    level4 = (
        row.get("레벨4", "")
        if row.get("레벨4", "") and pd.notna(row.get("레벨4", ""))
        else ""
    )
    std_unit = (
        row.get("표준 단위", "")
        if row.get("표준 단위", "") and pd.notna(row.get("표준 단위", ""))
        else ""
    )

    q1 = f"{level2} {level3} {level4} {std_unit}".strip()
    if q1:
        queries.append({"type": "simple_2_3_4", "query": q1})

    # 2) 레벨3,4,표준단위 조합
    q2 = f"{level3} {level4} {std_unit}".strip()
    if q2 and q2 != q1:
        queries.append({"type": "simple_3_4", "query": q2})

    # 3) LLM을 통한 쿼리 생성
    system_prompt = """당신은 열차 제작을 위한 사양서 문서에서 부품 또는 성능에 대한 요구사항을 찾기 위한 검색 질의를 생성하는 전문가입니다.

문서는 다음과 같이 **계층적 목차 구조**를 가지고 있으며, 각 항목은 사양서 안에서 다양한 표현 방식으로 기술되어 있습니다.
예시:
문서 목차 구조 : 4. 기술사항 > 4.3 차량특성 > 4.3.8 제어공기 압력  
내용: "주공기(MR) 883kPa(9kgf/cm²), 제동압력(BC) 490kPa 이하, 각종 공압제어장치 490kPa(동작범위 392～588kPa)"

사용자가 찾는 항목은 다음과 같은 구조로 제공됩니다:
예시 입력:  
- 레벨1: 운영 조건  
- 레벨2: 운영 환경  
- 레벨3: 최저 온도  
- 표준단위: ℃  

당신의 역할은 다음 조건에 맞는 **5개의 검색 질의(Query)**를 생성하는 것입니다:

1. 사용자가 찾고자 하는 정보가 문서 내 표현 방식과 다를 수 있으므로 **전문 용어, 유의어, 다양한 기술적 표현**을 활용해주세요.
2. **문장형 쿼리 또는 핵심어 조합형 쿼리**가 혼합되어도 좋습니다 (예: "최저 온도는?" / "운영 조건 온도 ℃").
3. **숫자 단위(예: mm, kPa, ℃ 등)도 중요한 검색 키워드**가 될 수 있으니 포함해주세요.
4. 최종 목적은 벡터 검색 시스템에서 해당 정보를 가장 잘 찾을 수 있는 질의어 세트를 생성하는 것입니다.

출력 예시:  
- "열차가 운행 가능한 최저 온도는?"  
- "기후 조건에서의 최저 온도는 몇 ℃인가요?"  
- "열차의 최소 운전 가능 온도"  
- "운영환경 기준 최저 기온 ℃"  
- "train minimum operating temperature in Celsius"
    """
    level1 = (
        row.get("레벨1", "")
        if row.get("레벨1", "") and pd.notna(row.get("레벨1", ""))
        else ""
    )
    user_prompt = f"""이제 아래 항목에 따라 5개의 검색 질의를 생성해주세요:
    레벨1: {level1}
    레벨2: {level2}
    레벨3: {level3} 
    레벨4: {level4}
    표준단위: {std_unit}
    쿼리는 순서대로 나열만 해주세요. 설명이나 번호는 필요 없습니다."""

    llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
    response = llm.predict(system_prompt + "\n\n" + user_prompt)
    # LLM 응답을 쿼리 목록으로 변환
    llm_queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
    for i, q in enumerate(llm_queries):
        queries.append({"type": f"llm_generated_{i+1}", "query": q})
    print("queries::", queries)
    return queries


# 빈 데이터프레임 초기화 함수
def initialize_empty_dataframe():
    # CSV에서 테이블 구조 로드
    table_df = load_table_data()
    result_df = table_df.copy()

    # 결과 열 추가 (미리 추가하여 데이터 타입 문제 방지)
    if "LLM응답" not in result_df.columns:
        result_df["LLM응답"] = ""
    if "참조문서" not in result_df.columns:
        result_df["참조문서"] = None
    if "참조문서목차" not in result_df.columns:
        result_df["참조문서목차"] = ""
    if "정답여부" not in result_df.columns:
        result_df["정답여부"] = False
    if "검색성공여부" not in result_df.columns:
        result_df["검색성공여부"] = False

    return result_df


# 세션 상태에 데이터프레임 초기화 (앱이 처음 실행될 때만)
if "result_df" not in st.session_state:
    st.session_state.result_df = initialize_empty_dataframe()

# 세션 상태에 벡터 DB 결과 초기화
if "vector_db_result" not in st.session_state:
    st.session_state.vector_db_result = None

# 사이드바 설정
with st.sidebar:
    st.header("설정")

    # 파일 업로드 버튼
    uploaded_file = st.file_uploader("사양서 파일 업로드 (DOCX)", type=["docx"])

    # LLM 모델 선택
    model_option = st.radio(
        "LLM 모델 선택", ["GPT-4", "Claude 3 Opus", "Claude 3 Sonnet"]
    )

    # 추출 시작 버튼
    start_button = st.button("사양 추출 시작", type="primary")

    # 초기화 버튼 추가
    reset_button = st.button("결과 초기화")

    # 추가 정보
    st.info(
        "이 앱은 RAG(Retrieval Augmented Generation)를 사용하여 문서에서 사양 정보를 추출합니다."
    )

# 메인 콘텐츠
if uploaded_file is not None:
    st.write("파일이 업로드되었습니다:", uploaded_file.name)

# 결과 초기화 처리
if reset_button:
    st.session_state.result_df = initialize_empty_dataframe()
    st.session_state.vector_db_result = None
    st.success("결과가 초기화되었습니다.")


# 데이터 프레임 영역 - 항상 표시
st.subheader("열차 사양 분석 항목")
result_placeholder = st.empty()
result_placeholder.dataframe(st.session_state.result_df, use_container_width=True)

# 성능 평가 버튼 영역
evaluate_placeholder = st.empty()

# 상태 표시 영역
status_placeholder = st.empty()

# 결과 다운로드 버튼 영역
download_placeholder = st.empty()


# 성능 평가 llm 처리
def evaluate_llm(pred, answer, answer_doc, ref_doc):
    llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
    ev_prompt = f"""다음은 정답과 정답의 근거 문서 입니다.
    정답: {pred}
    정답 출처: {answer_doc}
    
    위 정답과 자동 추출 결과 값이 같은지 true/false로 답해주세요. 숫자 값이나 단위, 설명 등 같은 의미를 가지는 경우 같은 값으로 판단합니다.
    또한 rag로 검색한 청크들 중 위 정답 출처 문서 내용이 있는지 확인하여 true/false로 답해주세요. 
    추출 결과: {answer}
    rag 검색한 참조 청크들: {ref_doc}

    답변 형식:
    정답 일치 여부: true/false
    청크 검색 성공 여부: true/false
    """
    response = llm.predict(ev_prompt)
    print("evaluate_llm::", response)

    answer_correct = (
        response.split("정답 일치 여부:")[1].split("청크 검색 성공 여부:")[0].strip()
        == "true"
    )
    search_success = response.split("청크 검색 성공 여부:")[1].strip() == "true"

    return answer_correct, search_success


# 성능 평가 처리
def evaluate_performance():
    # 정답여부와 검색성공여부 한번에 계산
    results = st.session_state.result_df.apply(
        lambda row: evaluate_llm(
            row["LLM응답"], row["정답"], row["정답 문서"], row["참조문서"]
        ),
        axis=1,
    )

    # 결과를 각 열에 할당
    st.session_state.result_df["정답여부"] = [result[0] for result in results]
    st.session_state.result_df["검색성공여부"] = [result[1] for result in results]

    # 데이터프레임 스타일링 함수
    def highlight_correct(row):
        if row["정답여부"]:
            return [
                "background-color: #CCFFCC" if col == "LLM응답" else ""
                for col in row.index
            ]
        else:
            return [
                "background-color: #FFCCCC" if col == "LLM응답" else ""
                for col in row.index
            ]

    # 스타일링 적용
    styled_df = st.session_state.result_df.style.apply(highlight_correct, axis=1)

    # 결과 표시
    result_placeholder.dataframe(styled_df, use_container_width=True)

    # 정답률 계산
    correct = st.session_state.result_df["정답여부"].sum()
    total = len(st.session_state.result_df)
    accuracy = correct / total if total > 0 else 0

    # 검색 재현율 계산
    hit = st.session_state.result_df["검색성공여부"].sum()
    recall = hit / total if total > 0 else 0

    st.session_state.evaluation_result = (
        f"정답률: {accuracy:.0%} ({correct}/{total}), "
        f"검색 재현율: {recall:.0%} ({hit}/{total})"
    )


# 분석 처리
if start_button and uploaded_file is not None:
    # 상태 메시지 표시
    status_placeholder.info("문서 처리를 시작합니다...")

    # 벡터 DB 생성 프로세스 시작
    try:
        chunks = convert_docx_to_chunks(uploaded_file)

        # OpenAI 임베딩 모델 초기화
        embeddings = OpenAIEmbeddings()
        print("embeddings 변수 초기화 완료")

        # Chroma 벡터 스토어 생성 - 공식 문서 방식대로
        vectorstore = Chroma(
            collection_name="langchain",
            embedding_function=embeddings,
            # persist_directory=persist_directory,
        )
        k = 5
        vectorstore.add_documents(chunks)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        # BM25 리트리버 생성
        bm25_retriever = BM25Retriever.from_documents(chunks)

        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )

        # 문서 청킹/벡터화 완료 후 데이터 추출 시작
        status_placeholder.success("문서 벡터화 완료. 사양 정보 추출 중...")

        total_rows = len(st.session_state.result_df)
        progress_bar = st.progress(0)

        for index, row in st.session_state.result_df.iterrows():
            # 진행 상태 표시
            progress = (index + 1) / total_rows
            progress_bar.progress(progress)

            status_placeholder.info(f"항목 {index+1}/{total_rows} 처리 중...")

            ensemble_docs = []
            queries = generate_queries(row)
            for query in queries:
                docs = ensemble.get_relevant_documents(query["query"])
                print("docs::", docs)
                ensemble_docs.extend(docs)
            # 중복 제거
            unique_docs = {}
            for doc in ensemble_docs:
                # doc 객체의 id 필드를 사용
                if hasattr(doc, "id"):
                    doc_key = doc.id
                else:
                    doc_key = doc.page_content
                unique_docs[doc_key] = doc
            ensemble_docs = list(unique_docs.values())
            # 빈 쿼리 제외
            filtered_queries = [
                query["query"] for query in queries if query["query"].strip()
            ]
            query_used = ", ".join(filtered_queries)
            answer_prompt = f"""다음은 열차 사양서의 여러 부분입니다:
{ensemble_docs}
위 문서에서 아래 질의들에 대한 답변과 참조 문서의 목차를 찾아주세요.  
1), 가) 등 세부 목차 구분자는 포함하되, **문서 제목이나 내용은 포함하지 마세요.** 아래 질의들은 모두 동일 항목에 대한 질문입니다.  
문서에 값이 명시되어 있다면 단위도 함께 표시해주세요.  

💡 **반드시 아래 예시처럼 간결한 형식**으로 응답해주세요:  
예시1)  
답변: 1000mm, 참조문서: 3.3 1)  
예시2)  
답변: 철도 규정에 따른다., 참조문서: 4.3.8 1)  
예시3)  
답변: 75dB, 참조문서: 4.3.11.2 1) 가)

❗참조문서는 **숫자, 1), 가)** 등 **목차 정보만 포함하고 제목이나 본문 내용은 절대 포함하지 마세요.**  
값을 찾을 수 없다면 '정보 없음'이라고 응답해주세요.
질문: '{query_used}'"""

            llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
            answer = llm.predict(answer_prompt)
            values = answer.split("답변:")[1].split(", 참조문서:")[0].strip()
            doc_index = answer.split("참조문서:")[-1].strip()
            print("답변::", values)
            print("참조문서::", doc_index)

            # 값 할당 시 데이터 타입 문제 방지
            st.session_state.result_df.loc[index, "LLM응답"] = str(values)
            st.session_state.result_df.loc[index, "참조문서목차"] = str(doc_index)

            # 객체를 직접 저장하면 문제가 발생할 수 있으므로 문자열로 변환
            st.session_state.result_df.loc[index, "참조문서"] = str(ensemble_docs)

            # 각 항목 처리 후 UI 업데이트
            result_placeholder.dataframe(
                st.session_state.result_df, use_container_width=True
            )

        progress_bar.empty()
        evaluate_performance()
        if st.session_state.evaluation_result:
            status_placeholder.success(st.session_state.evaluation_result)

        # 다운로드 버튼 활성화
        csv = st.session_state.result_df.to_csv(index=False)
        download_placeholder.download_button(
            label="CSV로 다운로드",
            data=csv,
            file_name="열차_사양_추출결과.csv",
            mime="text/csv",
        )

    except Exception as e:
        status_placeholder.error(f"처리 중 오류 발생: {str(e)}")

elif start_button and uploaded_file is None:
    st.error("먼저 사양서 파일을 업로드해주세요.")
