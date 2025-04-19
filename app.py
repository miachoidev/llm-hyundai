import streamlit as st
import pandas as pd
import os
from chunk_document import process_document
from retrive import load_vectorstore, load_bm25_retriever
import tempfile
from rag_llm import llm_response

# 페이지 설정
st.set_page_config(
    page_title="열차 사양서 분석기",
    page_icon="🚄",
    layout="wide"
)

# 제목 및 설명
st.title("열차 사양서 자동 분석 시스템")
st.markdown("열차 제작 공고사양서를 업로드하면 주요 스펙을 자동으로 추출하여 표로 보여줍니다.")

# CSV 파일에서 테이블 데이터 로드
# @st.cache_data - 캐시 제거 (개발 중 변경사항 즉시 확인 위함)
def load_table_data():
    try:
        # data 폴더에서 CSV 파일 로드
        csv_path = os.path.join("data", "hd_table.csv")
        df = pd.read_csv(csv_path, encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"CSV 파일 로드 중 오류 발생: {e}")
        # 오류 발생 시 기본 데이터프레임 반환
        return pd.DataFrame({
            "레이블1": ["오류", "CSV 파일을 찾을 수 없습니다"],
            "레이블2": ["", ""],
            "레이블3": ["", ""],
            "레이블4": ["", ""],
            "표준단위": ["", ""]
        })

# 빈 데이터프레임 초기화 함수
def initialize_empty_dataframe():
    # CSV에서 테이블 구조 로드
    table_df = load_table_data()
    
    # 결과 데이터프레임 구성
    # 원본 항목 유지하면서 값 칼럼 추가
    result_df = table_df.copy()
    
    return result_df

# 세션 상태에 데이터프레임 초기화 (앱이 처음 실행될 때만)
if 'result_df' not in st.session_state:
    st.session_state.result_df = initialize_empty_dataframe()

# 세션 상태에 벡터 DB 결과 초기화
if 'vector_db_result' not in st.session_state:
    st.session_state.vector_db_result = None

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    
    # 파일 업로드 버튼
    uploaded_file = st.file_uploader("사양서 파일 업로드 (DOCX)", type=["docx"])
    
    # LLM 모델 선택
    model_option = st.radio(
        "LLM 모델 선택",
        ["GPT-4", "Claude 3 Opus", "Claude 3 Sonnet"]
    )
    
    # 추출 시작 버튼
    start_button = st.button("사양 추출 시작", type="primary")
    
    # 초기화 버튼 추가
    reset_button = st.button("결과 초기화")
    
    # 저장소 상태 확인 버튼
    check_db_button = st.button("데이터 저장소 상태 확인")
    
    # 추가 정보
    st.info("이 앱은 RAG(Retrieval Augmented Generation)를 사용하여 문서에서 사양 정보를 추출합니다.")

# 메인 콘텐츠
if uploaded_file is not None:
    st.write("파일이 업로드되었습니다:", uploaded_file.name)

# 결과 초기화 처리
if reset_button:
    st.session_state.result_df = initialize_empty_dataframe()
    st.session_state.vector_db_result = None
    st.success("결과가 초기화되었습니다.")

# 데이터 프레임 영역 - 항상 표시
st.subheader("열차 사양 분석 결과")
result_placeholder = st.empty()
result_placeholder.dataframe(st.session_state.result_df, use_container_width=True)

# 진행 상태 표시 영역
progress_placeholder = st.empty()

# 결과 다운로드 버튼 영역
download_placeholder = st.empty()

# 데이터 저장소 상태 확인
if check_db_button:
    status_container = st.container()
    
    with status_container:
        st.subheader("데이터 저장소 상태")
        
        # 기본 저장 경로 확인
        default_vector_path = os.path.join(tempfile.gettempdir(), 'chroma_db')
        default_bm25_path = os.path.join(tempfile.gettempdir(), 'bm25_retriever')
        
        # 세션에 저장된 경로 확인
        if 'vector_db_result' in st.session_state and st.session_state.vector_db_result:
            vector_path = st.session_state.vector_db_result.get('persist_directory', default_vector_path)
            bm25_path = st.session_state.vector_db_result.get('bm25_directory', default_bm25_path)
        else:
            vector_path = default_vector_path
            bm25_path = default_bm25_path
        
        # 상태 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**벡터 데이터베이스:**")
            if os.path.exists(vector_path):
                try:
                    vectorstore = load_vectorstore(vector_path)
                    st.success(f"✅ 벡터 DB가 존재합니다: {vector_path}")
                    # 추가 정보 표시 (가능하다면)
                    try:
                        collection = vectorstore._collection
                        count = collection.count()
                        st.info(f"저장된 문서 수: {count}")
                    except:
                        pass
                except Exception as e:
                    st.error(f"❌ 벡터 DB 로드 실패: {str(e)}")
            else:
                st.error(f"❌ 벡터 DB 경로가 존재하지 않습니다: {vector_path}")
        
        with col2:
            st.write("**BM25 리트리버:**")
            bm25_file_path = os.path.join(bm25_path, 'bm25_retriever.pkl')
            if os.path.exists(bm25_file_path):
                try:
                    bm25_retriever = load_bm25_retriever(bm25_path)
                    st.success(f"✅ BM25 리트리버가 존재합니다")
                    # 파일 크기 표시
                    file_size = os.path.getsize(bm25_file_path) / (1024 * 1024)  # MB로 변환
                    st.info(f"파일 크기: {file_size:.2f} MB")
                except Exception as e:
                    st.error(f"❌ BM25 리트리버 로드 실패: {str(e)}")
            else:
                st.error(f"❌ BM25 리트리버 파일이 존재하지 않습니다: {bm25_file_path}")
        
        # 저장 경로 표시
        st.write("**저장 경로:**")
        st.code(f"벡터 DB: {vector_path}\nBM25: {bm25_path}")

# 분석 처리
if start_button and uploaded_file is not None:
    # 진행 상태 표시 초기화
    progress_bar = progress_placeholder.progress(0)
    status_text = st.empty()
    status_text.info("문서 처리를 시작합니다...")
    
    # 벡터 DB 생성 프로세스 시작
    try:
        # 테스트용 설정 - 벡터화 과정 건너뛰기
        skip_vectorization = False  # 벡터화 과정 스킵 (이미 벡터 DB가 있을 경우)
        
        if skip_vectorization:
            # 벡터화 스킵, 기존 벡터 DB 사용
            status_text.info("기존 벡터 DB를 사용합니다...")
            # 여기에 기존 벡터 DB 경로 설정 (테스트용)
            persist_directory = os.path.join(tempfile.gettempdir(), 'chroma_db')  # 기본 저장 위치
            vector_db_result = {
                'status': 'completed', 
                'progress': 1.0,
                'message': '기존 벡터 DB 사용 중',
                'persist_directory': persist_directory
            }
            progress_bar.progress(1.0)
            st.session_state.vector_db_result = vector_db_result
        else:
            # 실제 처리 모드
            vector_db_result = process_document(uploaded_file)
            progress_bar.progress(vector_db_result['progress'])
            status_text.info(vector_db_result['message'])
            
            # 벡터 DB 결과 저장 (chunks 객체 삭제)
            if 'chunks' in vector_db_result:
                # chunks는 직렬화하기 어려워 세션 상태에 저장하지 않고 개수만 저장
                chunk_count = vector_db_result['chunk_count']
                del vector_db_result['chunks']
            
            st.session_state.vector_db_result = vector_db_result
            
        # 문서 청킹/벡터화 완료 후 데이터 추출 시작
        status_text.success("문서 벡터화 완료. 사양 정보 추출 중...")
        
        # 여기서 RAG와 LLM을 사용하여 내용을 추출하게 됩니다
        result_df = llm_response(result_placeholder=result_placeholder, vector_db_result=st.session_state.vector_db_result)
        num_rows = len(result_df)
        st.session_state.result_df = result_df.copy()
        
        status_text.success('사양 정보 추출 완료!')
        
        # 다운로드 버튼 활성화
        csv = st.session_state.result_df.to_csv(index=False)
        download_placeholder.download_button(
            label="CSV로 다운로드",
            data=csv,
            file_name="열차_사양_추출결과.csv",
            mime="text/csv",
        )
        
    except Exception as e:
        status_text.error(f"처리 중 오류 발생: {str(e)}")
        
elif start_button and uploaded_file is None:
    st.error("먼저 사양서 파일을 업로드해주세요.")
