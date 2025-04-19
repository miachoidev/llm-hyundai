import os
import tempfile
import pickle
from typing import List, Dict, Any, Union, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def load_vectorstore(persist_directory=None):
    """
    벡터 스토어를 로드하는 함수
    
    Args:
        persist_directory: 벡터 스토어가 저장된 디렉토리 경로
        
    Returns:
        로드된 Chroma 벡터 스토어 객체
    """
    if persist_directory is None:
        # 기본 저장 위치 사용
        persist_directory = os.path.join(tempfile.gettempdir(), 'chroma_db')
    
    # 해당 디렉토리에 벡터 스토어가 존재하는지 확인
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"벡터 스토어 디렉토리를 찾을 수 없습니다: {persist_directory}")
    
    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()
    
    # 벡터 스토어 로드
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vectorstore

def similarity_retrieval(vectorstore, query: str, k: int = 5):
    """
    단순 유사도 검색을 수행하는 함수
    
    Args:
        vectorstore: Chroma 벡터 스토어 객체
        query: 검색 쿼리
        k: 검색할 문서 수
        
    Returns:
        검색된 문서 리스트
    """
    docs = vectorstore.similarity_search(query, k=k)
    return docs

def load_bm25_retriever(bm25_directory=None):
    """
    BM25 리트리버를 로드하는 함수
    
    Args:
        bm25_directory: BM25 리트리버가 저장된 디렉토리 경로
        
    Returns:
        로드된 BM25Retriever 객체
    """
    if bm25_directory is None:
        # 기본 저장 위치 사용
        bm25_directory = os.path.join(tempfile.gettempdir(), 'bm25_retriever')
    
    # 해당 디렉토리에 리트리버가 존재하는지 확인
    bm25_path = os.path.join(bm25_directory, 'bm25_retriever.pkl')
    backup_path = os.path.join(bm25_directory, 'bm25_retriever_backup.pkl')
    
    # 기본 파일 또는 백업 파일에서 로드 시도
    for file_path in [bm25_path, backup_path]:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    bm25_retriever = pickle.load(f)
                print(f"BM25 리트리버 로드 성공: {file_path}")
                return bm25_retriever
            except Exception as e:
                print(f"BM25 리트리버 로드 실패 ({file_path}): {str(e)}")
    
    # 백업 문서에서 재생성 시도
    try:
        # chunk_document.py에서 load_documents_backup 함수 임포트
        from chunk_document import load_documents_backup
        
        # 백업 문서 로드
        document_chunks = load_documents_backup()
        
        if document_chunks:
            print(f"백업 문서에서 BM25 리트리버 재생성 시도 ({len(document_chunks)} 청크)")
            # BM25 리트리버 생성
            bm25_retriever = BM25Retriever.from_documents(document_chunks)
            
            # 새로 생성한 리트리버 저장
            os.makedirs(bm25_directory, exist_ok=True)
            with open(bm25_path, 'wb') as f:
                pickle.dump(bm25_retriever, f)
            print(f"재생성한 BM25 리트리버 저장 완료: {bm25_path}")
            
            # 백업 파일도 생성
            with open(backup_path, 'wb') as f:
                pickle.dump(bm25_retriever, f)
            print(f"재생성한 BM25 리트리버 백업 저장 완료: {backup_path}")
            
            return bm25_retriever
        else:
            print("백업 문서를 찾을 수 없어 BM25 리트리버를 재생성할 수 없습니다.")
    except Exception as e:
        print(f"백업 문서에서 BM25 리트리버 재생성 실패: {str(e)}")
    
    raise FileNotFoundError(f"BM25 리트리버 파일을 찾을 수 없고 재생성에 실패했습니다: {bm25_path}")

def bm25_retrieval(bm25_retriever, query: str, k: int = 5):
    """
    BM25 검색을 수행하는 함수
    
    Args:
        bm25_retriever: BM25Retriever 객체
        query: 검색 쿼리
        k: 검색할 문서 수
        
    Returns:
        검색된 문서 리스트
    """
    # k 매개변수를 사용하여 문서 수 제한
    bm25_retriever.k = k
    docs = bm25_retriever.get_relevant_documents(query)
    return docs

def mmr_retrieval(vectorstore, query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.7):
    """
    MMR(Maximum Marginal Relevance) 검색을 수행하는 함수
    - 다양성과 관련성의 균형을 맞춘 검색 방식
    
    Args:
        vectorstore: Chroma 벡터 스토어 객체
        query: 검색 쿼리
        k: 최종 검색할 문서 수
        fetch_k: 초기 후보군 크기
        lambda_mult: 다양성-관련성 균형 조절 매개변수 (0~1, 높을수록 관련성 중시)
        
    Returns:
        검색된 문서 리스트
    """
    docs = vectorstore.max_marginal_relevance_search(
        query, 
        k=k, 
        fetch_k=fetch_k, 
        lambda_mult=lambda_mult
    )
    return docs

def contextual_compression_retrieval(vectorstore, query: str, k: int = 5, temperature: float = 0.0):
    """
    컨텍스트 압축 검색을 수행하는 함수
    - LLM을 사용해 관련성 높은 정보만 추출
    
    Args:
        vectorstore: Chroma 벡터 스토어 객체
        query: 검색 쿼리
        k: 검색할 문서 수
        temperature: LLM 생성 온도 (낮을수록 결정적)
        
    Returns:
        검색된 문서 리스트
    """
    # 기본 검색기 초기화
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    # LLM 초기화
    llm = ChatOpenAI(temperature=temperature)
    
    # 컨텍스트 압축기 초기화
    compressor = LLMChainExtractor.from_llm(llm)
    
    # 압축 검색기 초기화
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # 검색 수행
    compressed_docs = compression_retriever.get_relevant_documents(query)
    return compressed_docs

def metadata_filter_retrieval(vectorstore, query: str, filter_dict, k: int = 5):
    """
    메타데이터 필터링을 적용한 검색을 수행하는 함수
    
    Args:
        vectorstore: Chroma 벡터 스토어 객체
        query: 검색 쿼리
        filter_dict: 필터링할 메타데이터 조건 딕셔너리
        k: 검색할 문서 수
        
    Returns:
        검색된 문서 리스트
    """
    docs = vectorstore.similarity_search(
        query,
        k=k,
        filter=filter_dict
    )
    return docs

def format_results(documents, include_metadata: bool = True):
    """
    검색 결과를 포맷팅하는 함수
    
    Args:
        documents: 검색된 문서 리스트
        include_metadata: 메타데이터 포함 여부
        
    Returns:
        포맷팅된 결과 리스트
    """
    results = []
    
    for i, doc in enumerate(documents):
        result = {
            "id": i+1,
            "content": doc.page_content,
        }
        
        if include_metadata:
            result["metadata"] = doc.metadata
            
            # 경로 정보가 있으면 문자열로 변환
            if "path" in doc.metadata and isinstance(doc.metadata["path"], list):
                result["path_str"] = " > ".join(doc.metadata["path"])
            
        results.append(result)
    
    return results

def create_mmr_retriever(vectorstore, fetch_k: int = 20, lambda_mult: float = 0.7, k: int = 5):
    """
    MMR 리트리버를 생성하는 함수
    
    Args:
        vectorstore: Chroma 벡터 스토어 객체
        fetch_k: 초기 후보군 크기
        lambda_mult: 다양성-관련성 균형 조절 매개변수 (0~1, 높을수록 관련성 중시)
        k: 검색할 문서 수
        
    Returns:
        MMR 리트리버 객체
    """
    # MMR 리트리버 설정
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        }
    )

def ensemble_retrieval(vectorstore, bm25_retriever, query: str, k: int = 5, weights: List[float] = None, use_mmr: bool = False):
    """
    앙상블 검색을 수행하는 함수 - 벡터 검색과 BM25 검색 결과를 결합
    
    Args:
        vectorstore: Chroma 벡터 스토어 객체
        bm25_retriever: BM25Retriever 객체
        query: 검색 쿼리
        k: 각 리트리버에서 검색할 문서 수
        weights: 각 리트리버의 가중치 (기본값: 모든 리트리버에 동일한 가중치)
        use_mmr: MMR 리트리버를 앙상블에 포함할지 여부
        
    Returns:
        검색된 문서 리스트
    """
    retrievers = []
    
    # 벡터 리트리버 생성
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrievers.append(vector_retriever)
    
    # BM25 리트리버 설정
    bm25_retriever.k = k
    retrievers.append(bm25_retriever)
    
    # MMR 리트리버 추가
    if use_mmr:
        mmr_retriever = create_mmr_retriever(vectorstore, k=k)
        retrievers.append(mmr_retriever)
    
    # 가중치 설정 - 리트리버 수에 맞게 자동 조정
    if weights is None:
        weights = [1.0 / len(retrievers)] * len(retrievers)
    elif len(weights) != len(retrievers):
        # 가중치 수가 리트리버 수와 다르면 자동 조정
        weights = [1.0 / len(retrievers)] * len(retrievers)
    
    # 앙상블 리트리버 생성
    ensemble = EnsembleRetriever(
        retrievers=retrievers,
        weights=weights
    )
    
    # 검색 수행
    docs = ensemble.get_relevant_documents(query)
    return docs

def retrieve_and_extract(query: str, vectorstore=None, bm25_retriever=None, retrieval_method: str = "simple", k: int = 5):
    """
    메인 검색 함수 - 주어진 쿼리로 검색을 수행하고 결과를 반환
    
    Args:
        query: 검색 쿼리
        vectorstore: 벡터 스토어 객체 (None이면 로드 시도)
        bm25_retriever: BM25 리트리버 객체 (None이고 bm25 또는 ensemble 방식이면 로드 시도)
        retrieval_method: 검색 방법 ("simple", "mmr", "compression", "filter", "bm25", "ensemble", "ensemble_mmr")
        k: 검색할 문서 수
        
    Returns:
        포맷팅된 검색 결과
    """
    try:
        # 벡터 스토어가 없고 필요한 경우 로드 시도
        if vectorstore is None and retrieval_method != "bm25":
            vectorstore = load_vectorstore()
        
        # BM25 리트리버가 없고 필요한 경우 로드 시도
        if bm25_retriever is None and (retrieval_method == "bm25" or retrieval_method == "ensemble"):
            bm25_retriever = load_bm25_retriever()
        
        # 선택한 방법에 따라 검색 수행
        if retrieval_method == "mmr":
            docs = mmr_retrieval(vectorstore, query, k=k)
        elif retrieval_method == "compression":
            docs = contextual_compression_retrieval(vectorstore, query, k=k)
        elif retrieval_method == "filter":
            # 기본 필터 - 레벨이 2 이하인 문서만 검색
            filter_dict = {"level": {"$lte": 2}}
            docs = metadata_filter_retrieval(vectorstore, query, filter_dict, k=k)
        elif retrieval_method == "bm25":
            docs = bm25_retrieval(bm25_retriever, query, k=k)
        elif retrieval_method == "ensemble":
            docs = ensemble_retrieval(vectorstore, bm25_retriever, query, k=k, use_mmr=False)
        elif retrieval_method == "ensemble_mmr":
            docs = ensemble_retrieval(vectorstore, bm25_retriever, query, k=k, use_mmr=True)
        else:  # "simple" 또는 기본값
            docs = similarity_retrieval(vectorstore, query, k=k)
        
        # 결과 포맷팅
        return format_results(docs)
        
    except Exception as e:
        print(f"검색 중 오류 발생: {str(e)}")
        return []
        
   