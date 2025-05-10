# README 이미지 안내

이 디렉토리에는 README.md에 필요한 다음 이미지 파일들을 저장해주세요:

1. **demo.png**: 메인 데모 스크린샷 (800px 너비 권장)
2. **demo_main.png**: 앱 메인 화면 (600px 너비 권장)
3. **demo_result.png**: 결과 화면 (600px 너비 권장)
4. **demo_evaluation.png**: 성능 평가 화면 (600px 너비 권장)
5. **performance.png**: 성능 평가 결과 그래프 (600px 너비 권장)
6. **architecture.png**: 시스템 아키텍처 다이어그램 (800px 너비 권장)

이미지들은 PNG 포맷으로 준비하시고, GitHub 저장소에 업로드하시면 README.md에 자동으로 반영됩니다.

## 시스템 아키텍처 다이어그램 설명

아키텍처 다이어그램(architecture.png)은 다음과 같은 컴포넌트를 포함해야 합니다:

1. 문서 입력 (DOCX 파일)
2. 문서 처리 및 청킹 (chunker.py)
3. 벡터 임베딩 생성 (OpenAI Embeddings)
4. 벡터 저장소 (ChromaDB)
5. 하이브리드 검색 (BM25 + 벡터 검색)
6. LLM 추론 (gpt-4o-mini)
7. 결과 표시 및 평가 (Streamlit UI)

각 컴포넌트는 화살표로 연결하여 데이터 흐름을 표시해주세요. 