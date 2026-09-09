# QRiousRAG: 모호한 질의응답에서 LLMs의 질의 정제를 통한 검색 증강 생성

"QRiousRAG: Retrieval Augmented Generation with Question-refinement using LLMs in ambiguous question answering"이라는 논문으로 2024 한국소프트웨어종합학술대회에 게재되었습니다. 
해당 논문은 RAG 시스템이 모호한 질문에 대해 일관되고 신뢰할 수 있는 답변을 생성하지 못하는 문제를 다룹니다.

Link: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12042294

## Overview
기존 RAG 시스템은 입력 질문에 답변 품질이 좌우되어 정보가 불완전하거나 여러 해석이 가능한 질문에 취약합니다. 이를 해결하기 위해 검색된 문서의 관련성을 LLM 자가 피드백으로 평가하는 모듈과, 평가를 통과한 문서와 원 질의를 바탕으로 정제된 후속 질문을 생성하는 모듈을 기존 RAG 파이프라인에 추가한 QRiousRAG 프레임워크를 제안하였습니다. 

오픈소스 임베딩 모델(NV-Embed-v2)과 생성 모델(Llama-3.1)을 직접 설치 및 양자화하여 실험 환경을 구축하였고, ASQA 벤치마크에서 기존 최고 성능 모델 대비 Disambig-F1(정확도) 46.9%, DR-Score(종합 점수) 41.3%로 가장 높은 성능을 기록하였습니다.

<img width="5400" height="8400" alt="QRious_Poster_page-0001" src="https://github.com/user-attachments/assets/a7b8724d-98e8-4972-8244-6e0e70dc0511" />
