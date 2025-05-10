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
import concurrent.futures

# Disable Chroma telemetry to avoid protobuf issues
import chromadb
# chromadb.Client = lambda **kwargs: chromadb.Client(telemetry_enabled=False, **kwargs)

# 페이지 설정
st.set_page_config(page_title="열차 사양서 분석기", page_icon="🚄", layout="wide")

# 제목 및 설명
st.header("🚄 현대로템 - 열차 공고사양 자동 분석(PoC)")
st.markdown(
    """📄 본 시스템은 PoC(Proof of Concept)용으로, 공고 사양서 기반 사양 추출 및 평가 자동화를 시연합니다.  
사전에 정답이 포함된 데이터셋으로 LLM 추론 및 검색 성능을 평가합니다.
열차 제작 공고 사양서를 업로드 후 모델을 선택하고 '사양 추출 시작' 버튼을 누르세요.  

🤖 사용모델: gpt-4o-mini

📊 평가:  
LLM 추론 성능 평가 : 정답 기준 F1 / EM(Exact Match)  
RAG 검색 성능 측정 : Recall@K 방식으로 측정  
"""
)

# 시스템 흐름도 추가
with st.expander("시스템 프로세스 흐름도 보기"):
    st.graphviz_chart("""
    digraph {
        node [shape=box, style=filled, color=lightblue, fontname="나눔고딕"];
        
        upload [label="공고 사양서 업로드"];
        chunk [label="사양문서 청킹 및 임베딩"];
        query [label="공고항목별 검색쿼리 확장"];
        ensemble_search [label="확장쿼리로 ensemble_search"];
        llm [label="검색한 청크 기반 LLM 답변 추출 요청"];
        eval [label="성능 평가 (정확도/검색 재현율)"];
        
        up;
        chunk -> query;
        query -> ensemble_search;
        ensemble_search -> llm;
        llm -> eval;
        
        {rank=same; upload chunk query ensemble_search}
    }
    """)

# model
model_option = "gpt-4o-mini"

# 데이터프레임에서 표시할 열 목록 정의
DISPLAY_COLUMNS = [
    "레벨1",
    "레벨2",
    "레벨3",
    "레벨4",
    "정답",
    "정답 목차",
    "LLM응답",
    "참조문서목차",
    "참조문서",
    "정답여부",
    "검색성공여부",
]


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
                "레벨1": ["오류", "CSV 파일을 찾을 수 없습니다"],
                "레벨2": ["", ""],
                "레벨3": ["", ""],
                "레벨4": ["", ""],
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

    llm = ChatOpenAI(temperature=0.2, model=model_option)
    response = llm.predict(system_prompt + "\n\n" + user_prompt)
    # LLM 응답을 쿼리 목록으로 변환
    llm_queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
    for i, q in enumerate(llm_queries):
        queries.append({"type": f"llm_generated_{i + 1}", "query": q})
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
    # model_option = st.radio(
    #     "LLM 모델 선택", ["gpt-4o-mini", "claude-3-5-haiku-20241022"]
    # )
    # 추출 시작 버튼
    start_button = st.button("사양 추출 시작", type="primary", use_container_width=True)

    # 초기화 버튼 추가
    reset_button = st.button("결과 초기화", use_container_width=True)

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
result_placeholder = st.empty()
# 사용자가 원하는 열만 표시 (데이터는 모두 유지)
result_placeholder.dataframe(
    st.session_state.result_df[DISPLAY_COLUMNS], use_container_width=True
)

# 성능 평가 버튼 영역
evaluate_placeholder = st.empty()

# 상태 표시 영역
status_placeholder = st.empty()

# 결과 다운로드 버튼 영역
download_placeholder = st.empty()


# 성능 평가 llm 처리
def evaluate_llm(gold_answer, gold_doc, pred, ref_doc):
    llm = ChatOpenAI(temperature=0.2, model=model_option)
    ev_prompt = f"""다음은 정답과 정답의 근거 문서 입니다.
    정답: {gold_answer}
    정답 출처: {gold_doc}
    
    위 정답과 자동 추출 결과 값이 같은지 true/false로 답해주세요. 숫자 값이나 단위, 설명 등 같은 의미를 가지는 경우 같은 값으로 판단합니다.
    또한 rag로 검색한 청크들 중 위 정답 출처 문서 내용이 있는지 확인하여 true/false로 답해주세요. 
    추출 결과: {pred}
    rag 검색한 참조 청크들: {ref_doc}

    답변 형식:
    정답 일치 여부: true/false
    청크 검색 성공 여부: true/false
    """
    response = llm.predict(ev_prompt)

    answer_correct = (
        response.split("정답 일치 여부:")[1].split("청크 검색 성공 여부:")[0].strip()
        == "true"
    )
    search_success = response.split("청크 검색 성공 여부:")[1].strip() == "true"

    return answer_correct, search_success


# 성능 평가 처리
def evaluate_performance():
    status_placeholder.info("성능 평가 진행 중...")

    # 병렬 처리용 함수 정의
    def process_evaluation(args):
        index, row = args
        try:
            answer_correct, search_success = evaluate_llm(
                row["정답"], row["정답 문서"], row["LLM응답"], row["참조문서"]
            )
            return index, answer_correct, search_success
        except Exception as e:
            print(f"항목 {index} 평가 중 오류 발생: {e}")
            return index, False, False

    # 병렬 처리 실행
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_evaluation, (index, row))
            for index, row in st.session_state.result_df.iterrows()
        ]

        # 결과를 저장할 딕셔너리 초기화
        results_dict = {"정답여부": {}, "검색성공여부": {}}

        # 결과가 완료되는 대로 처리
        for future in concurrent.futures.as_completed(futures):
            try:
                index, answer_correct, search_success = future.result()
                results_dict["정답여부"][index] = answer_correct
                results_dict["검색성공여부"][index] = search_success
            except Exception as e:
                print(f"평가 결과 처리 중 오류 발생: {e}")

    # 결과를 데이터프레임에 일괄 할당
    for index, is_correct in results_dict["정답여부"].items():
        st.session_state.result_df.loc[index, "정답여부"] = (
            "true" if is_correct else "false"
        )

    for index, is_success in results_dict["검색성공여부"].items():
        st.session_state.result_df.loc[index, "검색성공여부"] = (
            "true" if is_success else "false"
        )

    # 스타일링 함수 정의
    def style_dataframe(df):
        # 정답여부와 검색성공여부에 스타일 적용
        styler = df.style.applymap(
            lambda v: (
                "background-color: #CCFFCC"
                if v == "true"
                else "background-color: #FFCCCC"
            ),
            subset=["정답여부", "검색성공여부"],
        )
        return styler

    # 스타일 적용하여 데이터프레임 표시
    styled_df = style_dataframe(st.session_state.result_df[DISPLAY_COLUMNS])
    result_placeholder.dataframe(styled_df, use_container_width=True)

    # 정답률 계산 (원래 불리언 값으로 계산해야 하므로 문자열을 다시 불리언으로 변환)
    correct = (st.session_state.result_df["정답여부"] == "true").sum()
    total = len(st.session_state.result_df)
    accuracy = correct / total if total > 0 else 0

    # 검색 재현율 계산
    hit = (st.session_state.result_df["검색성공여부"] == "true").sum()
    recall = hit / total if total > 0 else 0

    st.session_state.evaluation_result = (
        f"정답률: {accuracy:.0%} ({correct}/{total}), "
        f"검색 재현율: {recall:.0%} ({hit}/{total})"
    )

    status_placeholder.success("성능 평가 완료!")


# 분석 처리
if start_button and uploaded_file is not None:
    # 상태 메시지 표시
    status_placeholder.info("문서 처리를 시작합니다...")

    # 벡터 DB 생성 프로세스 시작
    try:
        chunks = convert_docx_to_chunks(uploaded_file)

        # OpenAI 임베딩 모델 초기화
        embeddings = OpenAIEmbeddings()

        # Chroma 벡터 스토어 생성 - 공식 문서 방식대로
        vectorstore = Chroma(
            collection_name="langchain",
            embedding_function=embeddings,
            # persist_directory=persist_directory,
        )
        k = 3
        vectorstore.add_documents(chunks)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        # BM25 리트리버 생성
        bm25_retriever = BM25Retriever.from_documents(chunks)

        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )

        # 문서 청킹/벡터화 완료 후 데이터 추출 시작
        status_placeholder.success("문서 벡터화 완료. 사양 정보 병렬 추출 중...")

        # 병렬 처리 함수 정의
        def process_row(args):
            index, row = args
            ensemble_docs = []
            queries = generate_queries(row)
            for query in queries:
                docs = ensemble.get_relevant_documents(query["query"])
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
            answer_prompt = f"""당신은 열차 제작 사양서의 전문가입니다. 다음은 열차 제작 사양서의 여러 부분입니다:
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

            llm = ChatOpenAI(temperature=0.2, model=model_option)
            answer = llm.predict(answer_prompt)
            values = answer.split("답변:")[1].split(", 참조문서:")[0].strip()
            doc_index = answer.split("참조문서:")[-1].strip()

            return index, values, doc_index, str(ensemble_docs)

        # 병렬 처리 실행
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(process_row, (index, row))
                for index, row in st.session_state.result_df.iterrows()
            ]

            # 결과가 완료되는 대로 처리
            for future in concurrent.futures.as_completed(futures):
                try:
                    index, values, doc_index, docs_str = future.result()

                    # 값 할당 시 데이터 타입 문제 방지
                    st.session_state.result_df.loc[index, "LLM응답"] = str(values)
                    st.session_state.result_df.loc[index, "참조문서목차"] = str(
                        doc_index
                    )
                    st.session_state.result_df.loc[index, "참조문서"] = docs_str

                    # 각 항목 처리 후 UI 업데이트
                    result_placeholder.dataframe(
                        st.session_state.result_df[DISPLAY_COLUMNS],
                        use_container_width=True,
                    )
                except Exception as e:
                    print(f"항목 {index} 처리 중 오류 발생: {e}")

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
