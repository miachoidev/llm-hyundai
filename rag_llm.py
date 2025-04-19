import streamlit as st
import pandas as pd
import concurrent.futures
from langchain_openai import ChatOpenAI
from retrive import (similarity_retrieval, bm25_retrieval, mmr_retrieval, 
                    format_results, load_vectorstore, load_bm25_retriever,
                    ensemble_retrieval)

def generate_queries(row):
    """여러 방식으로 쿼리 생성"""
    queries = []
    
    # 1) 레벨2,3,4,표준단위 조합
    q1 = f"{row.get('레벨2', '')} {row.get('레벨3', '')} {row.get('레벨4', '')} {row.get('표준 단위', '')}".strip()
    if q1: queries.append({"type": "simple_2_3_4", "query": q1})
    
    # 2) 레벨3,4,표준단위 조합
    q2 = f"{row.get('레벨3', '')} {row.get('레벨4', '')} {row.get('표준 단위', '')}".strip()
    if q2 and q2 != q1: queries.append({"type": "simple_3_4", "query": q2})
    
    # 3) LLM을 통한 쿼리 생성
    system_prompt = """당신은 열차 사양서 문서에서 특정 정보를 찾기 위한 검색 쿼리를 생성하는 전문가입니다.
    문서는 다음과 같은 구조로 저장되어 있습니다: 구조만 참고하세요.
    ```
    {
      "id": 66,
      "content": "4.3.8  제어공기 압력\n   주공기(MR) 883kPa(9kgf/cm2),  제동압력(BC) 490kPa(5kgf/cm2) 이하, 각종 공압제어장치 490kPa(5kgf/cm2)(동작범위 392～588kPa(4 ～ 6㎏f/㎠))",
      "metadata": {
        "heading": "4.3.8  제어공기 압력",
        "level": 3,
        "path": [
          "4. 기술사항",
          "4.3  차량특성",
          "4.3.8  제어공기 압력"
        ]
      }
    }
    ```
    문서는 목차 구조를 가지고 있으며, 찾고자 하는 항목명과 문서의 용어가 다를 수 있습니다.
    벡터 검색 방식으로 정보를 찾을 수 있도록 5개의 서로 다른 검색 질의를 생성해주세요."""
    
    user_prompt = f"""다음 항목에 대한 벡터 검색용 질의문 5개를 생성해주세요. 동의어나 유사어로 대체하거나 전문적인 용어로 질의가능합니다. 관련있는 문서가 잘 나오도록 해주세요.:
    레벨1: {row.get('레벨1', '')} 레벨2: {row.get('레벨2', '')} 레벨3: {row.get('레벨3', '')} 레벨4: {row.get('레벨4', '')} {row.get('표준단위', '')}
    쿼리는 순서대로 나열만 해주세요. 설명이나 번호는 필요 없습니다."""
    
    llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
    response = llm.predict(system_prompt + "\n\n" + user_prompt)
    print("response::", response)
    # LLM 응답을 쿼리 목록으로 변환
    llm_queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
    for i, q in enumerate(llm_queries):
        queries.append({"type": f"llm_generated_{i+1}", "query": q})
    
    return queries

def multi_search(queries, vectorstore):
    print("multi_search 함수 호출")
    """여러 검색 방법으로 검색 실행"""
    all_results = []
    
    # BM25 리트리버 로드
    try:
        bm25_directory = None
        if 'vector_db_result' in st.session_state and st.session_state.vector_db_result and 'bm25_directory' in st.session_state.vector_db_result:
            bm25_directory = st.session_state.vector_db_result['bm25_directory']
        
        bm25_retriever = load_bm25_retriever(bm25_directory)
        print("BM25 리트리버 로딩 완료")
        use_ensemble = True
    except Exception as e:
        print(f"BM25 리트리버 로드 중 오류: {str(e)}")
        print("BM25 리트리버 없이 계속 진행합니다. 벡터 검색과 MMR 검색을 사용합니다.")
        bm25_retriever = None
        use_ensemble = False
    
    for query_item in queries:
        query = query_item["query"]
        query_type = query_item["type"]
        
        if use_ensemble:
            # MMR이 포함된 앙상블 검색 (벡터 + BM25 + MMR)
            try:
                ensemble_mmr_docs = ensemble_retrieval(vectorstore, bm25_retriever, query, k=5, use_mmr=True)
                for doc in ensemble_mmr_docs:
                    all_results.append({
                        "query": query,
                        "query_type": query_type,
                        "search_method": "ensemble_mmr",
                        "doc": doc
                    })
            except Exception as ensemble_err:
                print(f"앙상블 검색 중 오류 발생: {str(ensemble_err)}")
                # 앙상블 검색 실패 시 대체 검색 사용
                use_ensemble = False
        
        # 앙상블 검색이 불가능하면 벡터 검색과 MMR 검색 사용
        if not use_ensemble:
            # 유사도 검색
            sim_docs = similarity_retrieval(vectorstore, query, k=3)
            for doc in sim_docs:
                all_results.append({
                    "query": query,
                    "query_type": query_type,
                    "search_method": "similarity",
                    "doc": doc
                })
            
            # MMR 검색
            mmr_docs = mmr_retrieval(vectorstore, query, k=3)
            for doc in mmr_docs:
                all_results.append({
                    "query": query,
                    "query_type": query_type,
                    "search_method": "mmr",
                    "doc": doc
                })
    
    # 검색 결과가 없는 경우 확인
    if not all_results:
        print("경고: 검색 결과가 없습니다. 검색 실패.")
        return []
    
    # 중복 제거 (같은 ID의 문서는 한 번만 포함)
    unique_docs = {}
    for result in all_results:
        doc_id = result["doc"].metadata.get("id", "unknown")
        # 우선순위: ensemble_mmr > mmr > similarity
        if doc_id not in unique_docs:
            unique_docs[doc_id] = result
        elif result["search_method"] == "ensemble_mmr":
            unique_docs[doc_id] = result
        elif result["search_method"] == "mmr" and unique_docs[doc_id]["search_method"] != "ensemble_mmr":
            unique_docs[doc_id] = result
    
    print(f"최종 검색 결과: {len(unique_docs)} 개의 고유 문서")
    return list(unique_docs.values())

def extract_answer(search_results, row):
    """검색 결과에서 답변 추출"""
    # 검색된 모든 문서 컨텐츠 결합
    doc_contents = "\n\n".join([f"[문서 {i+1}]\n{r['doc'].page_content}" for i, r in enumerate(search_results)])
    
    # 검색에 사용된 쿼리 정보도 포함
    query_info = "\n".join([f"- {r['query_type']}: {r['query']}" for r in search_results[:5]])
    
    row_str = f" {row.get('레벨2', '')} {row.get('레벨3', '')} {row.get('레벨4', '')} {row.get('표준단위', '')}"
    
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    answer_prompt = f"""다음은 열차 사양서의 여러 부분입니다:
    
    {doc_contents}
    
    위 문서에서 '{row_str}' 항목에 해당하는 값을 찾아서 알려주세요.
    값만 간결하게 응답해주세요. 단위가 있다면 함께 표시해주세요.
    값을 찾을 수 없다면 '정보 없음'이라고 응답해주세요."""
    
    answer = llm.predict(answer_prompt)
    
    # 검색에 사용된 문서와 쿼리 정보 수집
    reference = ", ".join([f"{r['doc'].metadata.get('heading', 'Unknown')}" for r in search_results[:3]])
    query_used = ", ".join([f"{r['query_type']}: {r['query']}" for r in search_results[:2]])
    
    return {
        "value": answer.strip(),
        "reference": reference,
        "query_used": query_used
    }

def process_row_advanced(row, vectorstore):
    """개선된 행 처리 함수"""
    # 1. 여러 방식의 쿼리 생성
    queries = generate_queries(row)
    
    # 2. 여러 검색 방법으로 검색
    search_results = multi_search(queries, vectorstore)
    
    # 3. 검색 결과로부터 답변 추출
    answer_data = extract_answer(search_results, row)
    
    # 결과 반환
    return {
        'index': row.name,
        'value': answer_data['value'],
        'reference': answer_data['reference'],
        'query_used': answer_data['query_used']
    }

def llm_response(result_placeholder=None, result_df=None, vector_db_result=None):
    """
    RAG 기반 LLM 응답 생성 함수
    
    Args:
        result_placeholder: Streamlit 데이터프레임 표시 영역
        result_df: 처리할 데이터프레임
        vector_db_result: 벡터 DB 결과 (persist_directory 포함)
    
    Returns:
        처리된 데이터프레임
    """
    # 데이터 프레임이 제공되지 않은 경우 세션 상태에서 가져오기
    if result_df is None and 'result_df' in st.session_state:
        result_df = st.session_state.result_df.copy()
    elif result_df is None:
        st.error("처리할 데이터프레임이 없습니다.")
        return None
    
    # 벡터 스토어 로드
    try:
        # 벡터 DB 결과에서 persist_directory 경로를 가져와 로드
        if vector_db_result and 'persist_directory' in vector_db_result:
            persist_directory = vector_db_result['persist_directory']
            vectorstore = load_vectorstore(persist_directory)
        elif 'vector_db_result' in st.session_state and st.session_state.vector_db_result and 'persist_directory' in st.session_state.vector_db_result:
            persist_directory = st.session_state.vector_db_result['persist_directory']
            vectorstore = load_vectorstore(persist_directory)
        else:
            vectorstore = load_vectorstore()
    except FileNotFoundError:
        st.error("벡터 스토어를 찾을 수 없습니다. 문서를 먼저 처리해주세요.")
        return result_df
    
    # 진행 상태 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.info("고급 사양 정보 추출 중...")
    
    # ThreadPoolExecutor를 사용한 병렬 처리
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_row_advanced, row, vectorstore): i 
                  for i, (_, row) in enumerate(result_df.iterrows())}
        total = len(futures)
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            # 진행 상황 업데이트
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.info(f"고급 사양 정보 추출 중... ({i+1}/{total})")
            
            # 결과 저장
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                st.error(f"항목 처리 중 오류 발생: {str(e)}")
    
    # 결과를 데이터프레임에 반영
    for result in results:
        idx = result['index']
        result_df.at[idx, 'LLM응답'] = result['value']
        result_df.at[idx, '청크'] = result['reference']
        result_df.at[idx, '사용된_쿼리'] = result.get('query_used', '')
    
    # UI에 결과 즉시 표시 (result_placeholder가 제공된 경우)
    if result_placeholder:
        result_placeholder.dataframe(result_df, use_container_width=True)
    
    status_text.success("고급 사양 정보 추출 완료!")
    
    return result_df
