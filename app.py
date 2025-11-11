
## 1. 라이브러리 로딩 ---------------------------------------------
import pandas as pd
import numpy as np
import os
import ast
import fitz  # PyMuPDF
from docx import Document
import random
import openai
import warnings
import json
import time
import difflib  # <= 추가
#PDF 생성
import typst
import json
import re
import gradio as gr
import tempfile


warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Annotated, Literal, Sequence, TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Markdown
from typing import TypedDict, List, Dict

# 질문 생성
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os, uuid
from collections import Counter

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import tempfile
import gradio as gr
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

## ---------------- 1단계 : 사전준비 ----------------------

# 0) 기업 핵심가치 및 인재상 입력 --
def get_company_context(company_name: str) -> str:
    context_map = {
        "삼성전자": "삼성전자는 '인재제일, 최고지향, 변화선도, 정도경영, 상생추구'의 핵심가치를 추구합니다.",
        "현대자동차": "현대자동차는 '도전적 실행, 고객 중심, 소통과 협력'을 중요시합니다.",
        "LG": "LG는 '정도경영, 고객가치창조, 인존경영'을 핵심 가치로 합니다.",
        "카카오": "카카오는 '상상력과 기술로 더 나은 세상'이라는 가치 아래 자율과 책임을 중시합니다.",
        "CJ": "CJ는 '정직, 열정, 창의'를 중심으로 ONLYONE 정신을 강조합니다."
    }
    return context_map.get(company_name, "이 기업은 도전, 협력, 책임감을 중요하게 여깁니다.")

# 1) 파일 입력 --------------------
def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    else:
        raise ValueError("지원하지 않는 파일 형식입니다. PDF 또는 DOCX만 허용됩니다.")

# 2) State 선언 --------------------


class InterviewState(TypedDict):
    # 고정 정보
    resume_text: str
    resume_summary: str
    resume_keywords: List[str]
    question_strategy: Dict[str, Dict]

    # 인터뷰 로그
    current_question: str
    current_answer: str
    current_strategy: str
    conversation: List[Dict[str, str]]
    evaluation : List[Dict[str, str]]
    next_step : str

    # 답변에 대한 평가 고도화
    need_reevaluation: bool  # 재평가 필요 여부
    reflection_reason: str   # reflection 사유

    #  인터뷰 진행 검토 고도화
    strategy_coverage: Dict[str, int]  # 전략별 질문 횟수
    deepening_count: int              # 심화 질문 횟수 제한
    company_name: str                 # 회사명 추가
    force_end: bool                    # 강제 종료 (필요시)
    deepening_target_index: int       # 심화 질문 대상 인덱스




# 3) resume 분석 --------------------
def analyze_resume(state: InterviewState) -> InterviewState:
    resume_text = state.get("resume_text", "")
    if not resume_text:
        raise ValueError("resume_text가 비어 있습니다. 먼저 텍스트를 추출해야 합니다.")

    llm = ChatOpenAI(model="gpt-4.1-mini")

    # 요약 프롬프트 구성
    summary_prompt = ChatPromptTemplate.from_template(
        '''당신은 이력서를 바탕으로 인터뷰 질문을 설계하는 AI입니다.
        다음 이력서 및 자기소개서 내용에서 질문을 뽑기 위한 중요한 내용을 10문장 정도로 요약을 해줘(요약시 ** 기호는 사용하지 말것):\n\n{resume_text}'''
    )
    formatted_summary_prompt = summary_prompt.format(resume_text=resume_text)
    summary_response = llm.invoke(formatted_summary_prompt)
    resume_summary = summary_response.content.strip()

    # 키워드 추출 프롬프트 구성
    keyword_prompt = ChatPromptTemplate.from_template(
        '''당신은 이력서를 바탕으로 인터뷰 질문을 설계하는 AI입니다.
        다음 이력서 및 자기소개서내용에서 질문을 뽑기 위한 중요한 핵심 키워드를 5~10개 추출해줘. 도출한 핵심 키워드만 쉼표로 구분해줘:\n\n{resume_text}'''
    )
    formatted_keyword_prompt = keyword_prompt.format(resume_text=resume_text)
    keyword_response = llm.invoke(formatted_keyword_prompt)

    parser = CommaSeparatedListOutputParser()
    resume_keywords = parser.parse(keyword_response.content)

    return {
        **state,
        "resume_summary": resume_summary,
        "resume_keywords": resume_keywords,
    }

# initial_state: InterviewState = {
#     "resume_text": resume_text,
#     "resume_summary": '',
#     "resume_keywords": [],
#     "question_strategy": {},

#     "current_question": '',
#     "current_answer": '',
#     "current_strategy": '',
#     "conversation": [],
#     "evaluation": [],
#     "next_step" : '',


#     "need_reevaluation": False,
#     "reflection_reason": '',


#     "strategy_coverage": {},           # 전략별 질문 횟수
#     "deepening_count": 0,              # 심화 질문 횟수 제한
#     "force_end": False,                 # 강제 종료 굳이 없어도 됨 삭제 가능
#     'deepening_target_index' : ''

# }



# 4) 질문 전략 수립 --------------------

def generate_question_strategy(state: InterviewState) -> InterviewState:
    # 여기에 코드를 완성합니다.

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    strategy_prompt = PromptTemplate(
        input_variables=["resume_summary", "resume_keywords"],
        template="""다음 이력서 요약과 키워드를 바탕으로 면접 질문 전략을 수립해주세요.

        이력서 요약:{resume_summary}

        주요 키워드:{resume_keywords}

        다음 3가지 분야에 대한 전략을 JSON 형식으로 작성해주세요:
        1. 경력 및 경험
        2. 동기 및 커뮤니케이션
        3. 논리적 사고

        각 분야마다 다음을 포함해주세요:
        - 질문 방향 (1-2문장)
        - 예시 질문 (2-3개)

        JSON 형식:
        {{
            "경력 및 경험": {{
                "질문_방향": "...",
                "예시_질문": ["질문1", "질문2", "질문3"]
            }},
            "동기 및 커뮤니케이션": {{
                "질문_방향": "...",
                "예시_질문": ["질문1", "질문2", "질문3"]
            }},
            "논리적 사고": {{
                "질문_방향": "...",
                "예시_질문": ["질문1", "질문2", "질문3"]
            }}
        }}
        """
    )

    strategy_chain = strategy_prompt | llm
    strategy_response = strategy_chain.invoke({
        "resume_summary": state["resume_summary"],
        "resume_keywords": ', '.join(state["resume_keywords"])
    }).content

    # JSON 파싱
    try:
        strategy_dict = json.loads(strategy_response)
    except:
        # JSON 파싱 실패 시 기본 전략
        strategy_dict = {
            "경력 및 경험": {
                "질문_방향": "지원자의 경력과 경험을 파악합니다.",
                "예시_질문": ["주요 경험에 대해 설명해주세요.", "프로젝트에서 어떤 역할을 했나요?"]
            },
            "동기 및 커뮤니케이션": {
                "질문_방향": "지원 동기와 소통 능력을 평가합니다.",
                "예시_질문": ["지원 동기는 무엇인가요?", "팀에서 협업한 경험을 말씀해주세요."]
            },
            "논리적 사고": {
                "질문_방향": "문제 해결 능력을 확인합니다.",
                "예시_질문": ["어려운 문제를 어떻게 해결했나요?", "의사결정 과정을 설명해주세요."]
            }
        }



    # return 코드는 제공합니다.
    return {
        **state,
        "question_strategy": strategy_dict
    }
# 4) 하나로 묶기 --------------------
def preProcessing_Interview(file_path: str, company_name: str = "") -> InterviewState:
    # 여기에 코드를 완성합니다.

    resume_text = extract_text_from_file(file_path)

    # state 초기화
    state = {
        "resume_text": resume_text,
        "resume_summary": '',
        "resume_keywords": [],
        "question_strategy": {},
        "current_question": '',
        "current_answer": '',
        "current_strategy": '',
        "conversation": [],
        "evaluation": [],
        "next_step": '',
        "company_name": company_name
    }

    # Resume 분석
    state = analyze_resume(state)

    # 질문 전략 수립
    state = generate_question_strategy(state)

    # 첫번째 질문 생성 (경력 및 경험의 첫 예시 질문)
    selected_question = state["question_strategy"]["경력 및 경험"]["예시_질문"][0]

    # return 코드는 제공합니다.
    return {
            **state,
            "current_question": selected_question,
            "current_strategy": "경력 및 경험"
            }
## ---------------- 2단계 : 면접 Agent ----------------------

# 1) 답변 입력 --------------------
def update_current_answer(state: InterviewState, user_answer: str) -> InterviewState:
    return {
        **state,
        "current_answer": user_answer.strip()
    }

# 2) 답변 평가 --------------------
def evaluate_answer(state: InterviewState) -> InterviewState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # 기업정보 가져오기
    company_name = state.get("company_name", "")
    company_context = get_company_context(company_name)

    # 질문과의 연관성 평가
    contect_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="""다음 면접 질문과 답변을 평가해주세요.

        질문: {question}
        답변: {answer}

        질문과의 연관성을 평가하고 "상", "중", "하" 중 하나만 먼저 쓴 후, 그 이유를 2-3문장으로 설명해주세요.

        평가 기준:
        - 상: 질문 의도에 정확히 부합하며 전반적인 내용을 명확히 다룸
        - 중: 질문과 관련은 있지만 핵심 포인트가 부분적으로 누락됨
        - 하: 질문과 관련이 약하거나 엉뚱한 내용 중심

        첫 단어로 "상", "중", "하" 중 하나를 쓰고, 그 다음에 이유를 작성하세요."""
    )

    # 답변의 구체성 평가
    detail_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="""다음 면접 질문과 답변을 평가해주세요.

        질문: {question}
        답변: {answer}

        답변의 구체성을 평가하고 "상", "중", "하" 중 하나만 먼저 쓴 후, 그 이유를 2-3문장으로 설명해주세요.

        평가 기준:
        - 상: 구체적인 사례, 수치, 방법론 등을 명확히 제시
        - 중: 일반적인 설명은 있으나 구체적인 예시가 부족
        - 하: 추상적이거나 모호한 표현만 사용

        첫 단어로 "상", "중", "하" 중 하나를 쓰고, 그 다음에 이유를 작성하세요."""
    )

    # 기업가치 부합도 평가
    value_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template=f"""
        아래 답변이 {company_name}의 인재상과 핵심가치({company_context})에
        얼마나 부합하는지를 객관적으로 분석하세요.

        당신은 {company_name}의 면접관이 아니라, 중립적인 외부 평가자입니다.

        질문: {{question}}
        답변: {{answer}}

        [평가 기준]
        - 상: 답변의 태도나 가치가 {company_name}의 핵심가치와 명확히 일치함
        - 중: 일부는 일치하지만 핵심가치가 충분히 드러나지 않음
        - 하: {company_name}의 가치와 거리가 있거나 반대되는 태도를 보임

        출력 형식: '상/중/하' + 이유 (2~3문장)
        """
    )

    contect_chain = contect_prompt | llm
    detail_chain = detail_prompt | llm
    value_chain = value_prompt | llm

    # 평가 실행
    contect_response = contect_chain.invoke({
        "question": state["current_question"],
        "answer": state["current_answer"]
    }).content.strip()

    detail_response = detail_chain.invoke({
        "question": state["current_question"],
        "answer": state["current_answer"]
    }).content.strip()
    value_response = value_chain.invoke({
        "question": state["current_question"],
        "answer": state["current_answer"]
    }).content.strip()

    # 등급 추출 (첫 단어 확인)
    contect_level = "중"
    if contect_response.startswith("상"):
        contect_level = "상"
    elif contect_response.startswith("하"):
        contect_level = "하"

    detail_level = "중"
    if detail_response.startswith("상"):
        detail_level = "상"
    elif detail_response.startswith("하"):
        detail_level = "하"

    value_level = "중"
    if value_response.startswith("상"):
        value_level = "상"
    elif value_response.startswith("하"):
        value_level = "하"

    eval_dict = {
        "질문과의_연관성": {
            "등급": contect_level,
            "근거": contect_response
        },
        "답변의_구체성": {
            "등급": detail_level,
            "근거": detail_response
        },
        "기업가치_부합도": {
            "등급": value_level,
            "근거": value_response
        }
    }

    print(f"[DEBUG] 평가 완료 - 연관성: {contect_level}, 구체성: {detail_level}, 가치부합: {value_level}")

    conversation = state["conversation"].copy()
    conversation.append({
        "질문": state["current_question"],
        "답변": state["current_answer"],
        "전략": state["current_strategy"]
    })

    evaluation = state["evaluation"].copy()
    evaluation.append(eval_dict)

    print(f"[DEBUG] conversation에 추가됨. 현재 길이: {len(conversation)}")

    return {
        **state,
        "conversation": conversation,
        "evaluation": evaluation
    }

# 분기 판단 함수 추가
def route_after_reflection(state: InterviewState) -> Literal["re_evaluate", "decide"]:
    if state.get("need_reevaluation", False):
        return "re_evaluate"
    else:
        return "decide"


# Reflection 노드: 평가 결과 검증
def reflect_evaluation(state: InterviewState) -> InterviewState:
    """
    LLM이 자동 평가한 결과에 대해 스스로 점검(reflect)
    평가 결과가 너무 관대하거나 부정확한 경우 판단
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # 기업정보 가져오기
    company_name = state.get("company_name", "")
    company_context = get_company_context(company_name)

    # 최근 평가 내용 가져오기
    if not state["evaluation"]:
        return {
            **state,
            "need_reevaluation": False
        }

    last_eval = state["evaluation"][-1]
    last_conv = state["conversation"][-1]

    reflect_prompt = PromptTemplate(
        input_variables=["company_name", "company_context", "question", "answer", "eval_relevance", "eval_detail"],
        template="""당신은 {company_name}의 면접관입니다.
        {company_context}

        다음 면접 평가 결과를 검토해주세요.

        질문: {question}
        답변: {answer}

        [현재 평가 결과]
        질문과의 연관성: {eval_relevance}
        답변의 구체성: {eval_detail}
        기업가치 부합도: {eval_value}

        위 평가가 적절한지 판단해주세요.

        다음 기준으로 판단하세요:
        1. 평가 등급이 답변 내용과 일치하는가?
        2. 너무 관대하게 평가하지 않았는가?
        3. 평가 근거가 논리적으로 타당한가?

        판단 결과를 다음 중 하나로 시작하세요:
        - "정상": 평가가 적절함
        - "재평가필요": 평가가 부정확하거나 관대함

        첫 단어로 "정상" 또는 "재평가필요"를 쓰고, 그 이유를 2-3문장으로 설명하세요."""
    )

    reflect_chain = reflect_prompt | llm
    reflect_response = reflect_chain.invoke({
        "company_name": state.get("company_name", ""),
        "company_context": get_company_context(state.get("company_name", "")),
        "question": last_conv['질문'],
        "answer": last_conv['답변'],
        "eval_relevance": f"{last_eval['질문과의_연관성']['등급']} - {last_eval['질문과의_연관성']['근거']}",
        "eval_detail": f"{last_eval['답변의_구체성']['등급']} - {last_eval['답변의_구체성']['근거']}",
        "eval_value": f"{last_eval['기업가치_부합도']['등급']} - {last_eval['기업가치_부합도']['근거']}"
    }).content.strip()

    need_reevaluation = reflect_response.startswith("재평가필요")

    print(f"[DEBUG] Reflection 결과: {'재평가 필요' if need_reevaluation else '정상'}")
    print(f"[DEBUG] {reflect_response}")

    return {
        **state,
        "need_reevaluation": need_reevaluation,
        "reflection_reason": reflect_response
    }


# 재평가 노드
def re_evaluate_answer(state: InterviewState) -> InterviewState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # 기업정보 가져오기
    company_name = state.get("company_name", "")
    company_context = get_company_context(company_name)

    last_conv = state["conversation"][-1]
    reflection_reason = state.get("reflection_reason", "")

    # 재평가 프롬프트 (더 엄격한 기준)
    reevaluate_prompt = PromptTemplate(
        input_variables=["question", "answer", "reflection_reason"],
        template="""다음 면접 질문과 답변을 재평가해주세요.

        질문: {question}
        답변: {answer}

        [이전 평가 문제점]
        {reflection_reason}

        더 엄격한 기준으로 재평가하세요:
        - 구체적인 사례나 수치가 없으면 "중" 이하
        - 질문의 핵심을 놓쳤으면 "하"
        - 추상적이거나 일반적인 답변은 "하"

        질문과의 연관성을 평가하고 "상", "중", "하" 중 하나만 먼저 쓴 후, 그 이유를 2-3문장으로 설명해주세요.

        평가 기준:
        - 상: 질문 의도에 정확히 부합하며 전반적인 내용을 명확히 다룸
        - 중: 질문과 관련은 있지만 핵심 포인트가 부분적으로 누락됨
        - 하: 질문과 관련이 약하거나 엉뚱한 내용 중심

        첫 단어로 "상", "중", "하" 중 하나를 쓰고, 그 다음에 이유를 작성하세요."""
    )

    detail_reevaluate_prompt = PromptTemplate(
        input_variables=["question", "answer", "reflection_reason"],
        template="""다음 면접 질문과 답변을 재평가해주세요.

        질문: {question}
        답변: {answer}

        [이전 평가 문제점]
        {reflection_reason}

        더 엄격한 기준으로 재평가하세요:
        - 구체적인 사례나 수치가 없으면 "중" 이하
        - 질문의 핵심을 놓쳤으면 "하"
        - 추상적이거나 일반적인 답변은 "하"

        답변의 구체성을 평가하고 "상", "중", "하" 중 하나만 먼저 쓴 후, 그 이유를 2-3문장으로 설명해주세요.

        평가 기준:
        - 상: 구체적인 사례, 수치, 방법론 등을 명확히 제시
        - 중: 일반적인 설명은 있으나 구체적인 예시가 부족
        - 하: 추상적이거나 모호한 표현만 사용

        첫 단어로 "상", "중", "하" 중 하나를 쓰고, 그 다음에 이유를 작성하세요."""
    )

    # 기업가치 부합도 재평가
    value_reevaluate_prompt = PromptTemplate(
        input_variables=["company_name", "company_context", "question", "answer", "reason"],
        template="""
        아래 답변이 {{company_name}}의 인재상과 핵심가치({{company_context}})에
        얼마나 부합하는지를 객관적으로 재평가하세요.

        당신은 면접관이 아니라 중립적인 외부 분석자이며,
        답변이 기업 가치와 일치하는 정도를 공정하게 판단해야 합니다.

        질문: {question}
        답변: {answer}

        [이전 평가 문제점]
        {reflection_reason}

        더 엄격한 기준으로 재평가하세요:
        - 일부만 부합하거나 가치가 약하게 드러나면 '중' 이하
        - 가치와 불일치하거나 반대되는 태도를 보이면 '하'

        기업의 인재상과 핵심가치에 적절한 지 평가하고 "상", "중", "하" 중 하나만 먼저 쓴 후, 그 이유를 2-3문장으로 설명해주세요.

        평가 기준:
        - 상: 답변 속 태도, 표현, 의사결정 방식이 기업의 핵심가치와 명확히 일치함
        - 중: 일부 표현은 일치하지만 전반적으로 가치관이 약하거나 모호하게 드러남
        - 하: 기업이 중시하는 가치와 반대되거나, 무관한 태도/사례 중심으로 서술함

        첫 단어로 "상", "중", "하" 중 하나를 쓰고, 그 다음에 이유를 작성하세요."""
    )

    relevance_chain = reevaluate_prompt | llm
    detail_chain = detail_reevaluate_prompt | llm
    value_chain = value_reevaluate_prompt | llm

    # 재평가 실행
    relevance_response = relevance_chain.invoke({
        "question": last_conv['질문'],
        "answer": last_conv['답변'],
        "reflection_reason": reflection_reason
    }).content.strip()

    detail_response = detail_chain.invoke({
        "question": last_conv['질문'],
        "answer": last_conv['답변'],
        "reflection_reason": reflection_reason
    }).content.strip()

    value_response = value_chain.invoke({
        "company_name": state.get("company_name", ""),
        "company_context": get_company_context(state.get("company_name", "")),
        "question": last_conv['질문'],
        "answer": last_conv['답변'],
        "reflection_reason": reflection_reason
    }).content.strip()

    # 등급 추출
    relevance_level = "중"
    if relevance_response.startswith("상"):
        relevance_level = "상"
    elif relevance_response.startswith("하"):
        relevance_level = "하"

    detail_level = "중"
    if detail_response.startswith("상"):
        detail_level = "상"
    elif detail_response.startswith("하"):
        detail_level = "하"

    value_level = "중"
    if detail_response.startswith("상"):
        value_level = "상"
    elif detail_response.startswith("하"):
        value_level = "하"

    # 재평가 결과로 교체
    new_eval_dict = {
        "질문과의_연관성": {
            "등급": relevance_level,
            "근거": relevance_response
        },
        "답변의_구체성": {
            "등급": detail_level,
            "근거": detail_response
        },
        "기업가치_부합도": {
            "등급": value_level,
            "근거": value_response
        }
    }

    # evaluation 리스트의 마지막 항목 교체
    evaluation = state["evaluation"].copy()
    evaluation[-1] = new_eval_dict

    print(f"[DEBUG] 재평가 완료 - 연관성: {relevance_level}, 구체성: {detail_level}")

    return {
        **state,
        "evaluation": evaluation,
        "need_reevaluation": False  # 재평가 완료 표시
    }


# 3) 인터뷰 진행 검토 --------------------

def decide_next_step(state: InterviewState) -> InterviewState:
    """
    전에 있던 사소한 문제점 커버
    심화 질문 눈적값 삭제
    최근 평가 + 전략 커버를 고려하고,
    전체 대화 5회 이상이면 종료

    """
    deepening_target_index = state.get("deepening_target_index", len(state.get("conversation", [])) - 1)
    conversation_count = len(state.get("conversation", []))
    deepening_count = state.get("deepening_count", 0)  # 전략 단위 심화 질문
    strategy_coverage = state.get("strategy_coverage", {})
    current_strategy = state.get("current_strategy", "기본전략")
    recent_eval = state.get("evaluation", [])[-1] if state.get("evaluation") else None
    grades = []

    if recent_eval:
        grades = [
            recent_eval["질문과의_연관성"]["등급"],
            recent_eval["답변의_구체성"]["등급"],
            recent_eval["기업가치_부합도"]["등급"]
        ]

        # 모두 중/상 → 전략별 커버 확인 후 다음 전략 또는 추가 질문
        if all(g in ["중", "상"] for g in grades):
            coverage = strategy_coverage.get(current_strategy, 0)
            max_questions = 2
            if coverage >= max_questions:
                next_step = "next_strategy"
                current_strategy = f"{current_strategy}_다음"
                strategy_coverage[current_strategy] = 0
                deepening_count = 0  # 전략 바뀌면 전략 단위 심화 초기화
            else:
                next_step = "additional_question"
                strategy_coverage[current_strategy] = coverage + 1

        # 하나라도 하 → 심화 질문
        elif any(g == "하" for g in grades) and conversation_count > 0:

            if deepening_count < 1:
                next_step = "deepening_question"
                deepening_count += 1
                deepening_target_index = len(state.get("conversation", [])) - 1

            else:
                # 전략별 1회 심화 질문 후 다음 전략
                next_step = "next_strategy"
                current_strategy = f"{current_strategy}_다음"
                strategy_coverage[current_strategy] = 0
                deepening_count = 0  # 전략 바뀌면 심화 초기화

        else:
            next_step = "additional_question"
    else:
        next_step = "additional_question"

    # 전체 대화 5회 이상이면 종료
    if conversation_count >= 5:
        next_step = "end"

    # 전체 전략 커버 완료 시 종료
    if state.get("question_strategy") and all(strategy_coverage.get(s, 0) > 0 for s in state["question_strategy"]):
        next_step = "end"

    print(f"[DEBUG] 대화 수: {conversation_count}, 전략: {current_strategy}, "
          f"커버: {strategy_coverage.get(current_strategy, 0)}, 최근 평가: {grades}, "
          f"심화 질문 횟수(전략 단위): {deepening_count}, "
          f"결정 단계: {next_step}")

    return {
        **state,
        "next_step": next_step,
        "current_strategy": current_strategy,
        "strategy_coverage": strategy_coverage,
        "deepening_count": deepening_count,
        "deepening_target_index": deepening_target_index,

    }


# 4) 질문 생성 --------------------

# ==========================================================
# 질문 생성 고도화 (주제 다양화 + 중복 회피) — InterviewState 스키마 준수
# ==========================================================
from typing import List, Dict
from collections import Counter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import random, re, uuid, os

# ---------------------------
# 기본 유틸
# ---------------------------
def _truncate(text: str, max_chars: int = 1200) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + " ..."
# ====== 질문 중복 체크 유틸 ======

def _normalize_q(q: str) -> str:
    """질문 문자열 정규화 (소문자 + 공백/특수문자 최소화)"""
    q = q.strip().lower()
    # 너무 과한 전처리는 필요 없고, 공백만 정리
    q = re.sub(r"\s+", " ", q)
    return q

def _similar_ratio(a: str, b: str) -> float:
    """difflib 기반 유사도 (0~1)"""
    return difflib.SequenceMatcher(None, _normalize_q(a), _normalize_q(b)).ratio()

def _collect_previous_questions(state: Dict) -> List[str]:
    """
    1) 지금까지 실제로 물어본 질문들
    2) question_strategy에 들어있는 예시 질문들
    을 모두 모아서 리스트로 반환
    """
    prev = []

    # 1) 대화 기록 질문들
    for turn in state.get("conversation", []):
        q = (turn.get("질문") or "").strip()
        if q:
            prev.append(q)

    # 2) 전략 예시 질문들
    for area, payload in (state.get("question_strategy") or {}).items():
        for q in payload.get("예시_질문", []):
            q = (q or "").strip()
            if q:
                prev.append(q)

    # 3) 현재 state에 이미 설정된 current_question도 포함
    cur_q = (state.get("current_question") or "").strip()
    if cur_q:
        prev.append(cur_q)

    # 중복 제거
    dedup = []
    seen = set()
    for q in prev:
        key = _normalize_q(q)
        if key not in seen:
            seen.add(key)
            dedup.append(q)
    return dedup

def _is_duplicate_question(new_q: str, prev_questions: List[str], threshold: float = 0.80) -> bool:
    """
    새로 생성된 질문이 이전 질문들과 유사한지 체크
    threshold 이상이면 '중복'으로 간주
    """
    for old in prev_questions:
        if _similar_ratio(new_q, old) >= threshold:
            return True
    return False


def _collect_question_corpus(state: Dict) -> List[Dict]:
    """
    question_strategy와 conversation에서 질문 코퍼스를 수집.
    각 항목은 {"id": str, "text": str, "meta": {...}} 형태.
    """
    corpus = []

    # 1) 전략의 예시 질문
    for area, payload in (state.get("question_strategy") or {}).items():
        for q in payload.get("예시_질문", []):
            corpus.append({
                "id": f"strategy::{area}::{uuid.uuid4().hex[:8]}",
                "text": q,
                "meta": {"source": "strategy", "area": area}
            })

    # 2) 과거 질문
    for item in state.get("conversation", []):
        if item.get("질문"):
            corpus.append({
                "id": f"conversation::{uuid.uuid4().hex[:8]}",
                "text": item["질문"],
                "meta": {"source": "conversation", "area": item.get("전략", "")}
            })

    return corpus

def _build_or_refresh_chroma(
    corpus: List[Dict],
    persist_dir: str = "./chroma_questions",
    collection_name: str = "interview_questions",
):
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    texts = [c["text"] for c in corpus]
    metadatas = [c["meta"] for c in corpus]
    ids = [c["id"] for c in corpus]

    vs = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    return vs

def _retrieve_similar_questions(vs: Chroma, query_text: str, k: int = 3, use_mmr: bool = True) -> List[str]:
    retriever = vs.as_retriever(
        search_type="mmr" if use_mmr else "similarity",
        search_kwargs=({"k": k, "fetch_k": 8} if use_mmr else {"k": k}),
    )
    try:
        docs = retriever.invoke(query_text)
    except AttributeError:
        docs = vs.similarity_search(query_text, k=k)
    return [d.page_content for d in docs][:k]

def _find_target_question_for_deepening(state: Dict) -> Dict:
    """
    심화 질문의 대상(가장 최근 or 지정 인덱스)을 반환.
    """
    conversation = state.get("conversation", [])
    if not conversation:
        return {"질문": "", "답변": ""}
    idx = state.get("deepening_target_index", len(conversation) - 1)
    if 0 <= idx < len(conversation):
        return conversation[idx]
    return conversation[-1]

# ---------------------------
# 키워드 다변화(로컬 계산만, state에 저장 X)
# ---------------------------
def _balance_keywords(keywords: List[str], top_n: int = 6) -> List[str]:
    """
    상위 키워드 편중 완화: 출현 빈도 상위 + 짧은 보조 키워드 일부 섞기.
    (state에는 저장하지 않고 호출 내에서만 사용)
    """
    if not keywords:
        return []
    counts = Counter(keywords)
    sorted_kw = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    core = [k for k,_ in sorted_kw[:top_n]]
    extras = [k for k,_ in sorted_kw[top_n:] if len(k) <= 6]
    random.shuffle(extras)
    mixed = core + extras[:2]
    # 중복 제거
    seen, balanced = set(), []
    for k in mixed:
        if k not in seen:
            balanced.append(k); seen.add(k)
    return balanced

# ---------------------------
# 메인 질문 생성 (스키마 준수: 새로운 키 저장하지 않음)
# ---------------------------
def generate_question(state: Dict) -> Dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

    # 기업가치 불러오기기
    company_name = state.get("company_name", "")
    company_context = get_company_context(company_name)

    next_step = state.get("next_step", "additional_question")
    is_deepening = (next_step == "deepening_question")
    # [추가] 금지할 기존 질문들 수집
    blocked_questions_list = _collect_previous_questions(state)
    blocked_questions_text = ""
    if blocked_questions_list:
        # 너무 많으면 최근 것 위주로 15개만
        recent_blocked = blocked_questions_list[-15:]
        blocked_questions_text = "\n".join(f"- {q}" for q in recent_blocked)


    # 1) 최근 대화/평가 요약
    recent_conversation = ""
    target_question_info = ""
    if is_deepening:
        target = _find_target_question_for_deepening(state)
        target_question_info = (
            f"[심화 질문 대상]\n"
            f"원래 질문: {target.get('질문', '')}\n"
            f"지원자 답변: {_truncate(target.get('답변', ''), 600)}\n"
        )
    else:
        if state.get("conversation"):
            last = state["conversation"][-1]
            recent_conversation = f"최근 질문: {last.get('질문','')}\n최근 답변: {last.get('답변','')}"

    recent_evaluation = ""
    if state.get("evaluation"):
        last_eval = state["evaluation"][-1]
        try:
            rel = last_eval["질문과의_연관성"]
            det = last_eval["답변의_구체성"]
            val = last_eval["기업가치_부합도"]
            recent_evaluation = (
                f"연관 평가: {rel.get('등급','-')} - {_truncate(rel.get('근거',''))}\n"
                f"구체성 평가: {det.get('등급','-')} - {_truncate(det.get('근거',''))}\n"
                f"가치부합도 평가: {val.get('등급','-')} - {_truncate(val.get('근거',''))}"
            )
        except Exception:
            pass

    # 2) 질문 코퍼스 + 유사 질문
    corpus = _collect_question_corpus(state)
    similar_block = ""
    balanced_keywords = _balance_keywords(state.get("resume_keywords", []) or [])

    # (주제 후보) — 최근에 쓴 주제 회피는 state를 건드리지 않고,
    # 단순히 balanced_keywords에서 무작위로 하나 뽑아 프롬프트 힌트로만 사용
    chosen_topic = random.choice(balanced_keywords) if balanced_keywords else ""

    if corpus:
        vs = _build_or_refresh_chroma(corpus)
        sampled = []
        if balanced_keywords:
            k = min(4, len(balanced_keywords))
            sampled = random.sample(balanced_keywords, k=k)

        if is_deepening:
            target = _find_target_question_for_deepening(state)
            query = "\n".join([
                f"[심화 대상 질문] {target.get('질문', '')}",
                f"[지원자 답변] {_truncate(target.get('답변', ''), 600)}",
                f"[현재 전략] {state.get('current_strategy','')}",
                f"[주제 후보] {chosen_topic}",
                f"[키워드 샘플] {', '.join(sampled)}"
            ])
        else:
            query = "\n".join([
                f"[현재 전략] {state.get('current_strategy','')}",
                f"[이력서 요약] {_truncate(state.get('resume_summary',''), 800)}",
                f"[주제 후보] {chosen_topic}",
                f"[키워드 샘플] {', '.join(sampled)}",
                f"[최근 답변] {(_truncate(state['conversation'][-1]['답변'], 400) if state.get('conversation') else '')}"
            ])

        similar_questions = _retrieve_similar_questions(vs, query, k=3, use_mmr=True)
        if similar_questions:
            bullet = "\n".join([f"- {q}" for q in similar_questions])
            similar_block = "\n[참고 유사 질문 (복사 금지, 참고만)]\n" + bullet + "\n"

    # 3) 프롬프트 (심화/일반 분기)
    DIVERSITY_BLOCK = (
        "[주제 다양화 지침]\n"
        "- 특정 한 경험에만 고정하지 말고, 이력서 키워드 중 하나로 질문 주제를 설정하세요.\n"
        "- 최근 대화에서 이미 다룬 표현/내용과 중복되지 않도록 의도/표현을 변형하지 말고 완전히 다른 각도를 취하세요.\n"
        "- 질문 전략(문제해결/리스크/협업/윤리/회고 등) 취지에 맞게 검증 포인트를 분명히 하세요.\n"
    )

    if is_deepening:
        question_prompt = PromptTemplate(
            input_variables=[
                "resume_summary", "company_context", "keywords","strategy","target_question_info",
                "recent_evaluation","similar_block","chosen_topic", "blocked_questions"
            ],
            template=(
                "다음 정보를 바탕으로 한국어 '심화 면접 질문'을 1개 생성하세요.\n\n"
                f"{DIVERSITY_BLOCK}\n"
                "이력서 요약:\n{resume_summary}\n"
                "기업 인재상 및 핵심가치: {company_context}\n"
                "주요 키워드: {keywords}\n"
                "현재 질문 전략: {strategy}\n\n"
                "원래 질문/지원자 답변 블록:\n{target_question_info}\n"
                "최근 자동평가 요약(누락/레드플래그 포함):\n{recent_evaluation}\n"
                "유사 질문(중복 금지 참고용):\n{similar_block}\n"
                "주제 후보: {chosen_topic}\n\n"
                "※ 아래 리스트에 있는 질문들과 의미·문장 구조가 유사한 질문은 절대 만들지 마세요.\n"
                "{blocked_questions}\n\n"
                "요구사항:\n"
                "1) 한 줄 질문만 출력. 설명/머리말/예시 금지.\n"
                "2) 방금 답변의 '모호/불충분/가정' 지점을 콕 집어 근거를 요구(베이스라인·대안비교·의사결정 기준·리스크/윤리 중 1~2개 초점).\n"
                "3) 본인 역할 vs 팀 역할을 구분해 검증.\n"
                "4) 유사 질문과 의미·표현 중복 금지(동의어 치환 포함).\n"
                "출력: 질문만 한 줄로."
            )
        )
    else:
        question_prompt = PromptTemplate(
            input_variables=[
                "resume_summary","company_context","keywords","strategy","recent_conversation",
                "recent_evaluation","similar_block","chosen_topic","blocked_questions"
            ],
            template=(
                "다음 정보를 바탕으로 한국어 '면접 질문'을 1개 생성하세요.\n\n"
                f"{DIVERSITY_BLOCK}\n"
                "이력서 요약:\n{resume_summary}\n\n"
                "기업 인재상 및 핵심가치: {company_context}\n"
                "주요 키워드: {keywords}\n"
                "현재 질문 전략: {strategy}\n"
                "최근 대화 요약:\n{recent_conversation}\n"
                "최근 자동평가 요약(누락/레드플래그 포함):\n{recent_evaluation}\n"
                "유사 질문(중복 금지 참고용):\n{similar_block}\n"
                "주제 후보: {chosen_topic}\n\n"
                "※ 아래 리스트에 있는 질문들과 의미·문장 구조가 유사한 질문은 절대 만들지 마세요.\n"
                "{blocked_questions}\n\n"
                "요구사항:\n"
                "1) 한 줄 질문만 출력(머리말/불릿/해설 금지).\n"
                "2) 사고과정·문제해결·트레이드오프·지표·리스크·윤리 등 중 하나를 깊게 검증.\n"
                "3) 최근 자동평가의 '누락 항목'을 보완(대안/리스크/역할명확성).\n"
                "4) 유사 질문과 표현·의도 중복 금지(동의어/구문변형도 회피).\n"
                "출력: 질문만 한 줄로."
            )
        )

    # 4) LLM 호출
    question_chain = question_prompt | llm
    prev_questions = blocked_questions_list
    new_q = None
    for attempt in range(3):  # 최대 3번 재시도
        response = question_chain.invoke({
            "resume_summary": _truncate(state.get("resume_summary",""), 800),
            "keywords": ', '.join(balanced_keywords),
            "strategy": state.get("current_strategy",""),
            "recent_conversation": recent_conversation,
            "recent_evaluation": recent_evaluation,
            "similar_block": similar_block,
            "target_question_info": target_question_info,
            "chosen_topic": chosen_topic,
            "company_context": company_context,
            "blocked_questions": blocked_questions_text,
        })

        q = response.content.strip().replace("\n", " ")
        q = re.sub(r"\s+", " ", q)
        if not q.endswith("?"):
            q += "?"
        if len(q) > 140:
            q = q[:137] + "..."

        # 유사도 체크
        if not _is_duplicate_question(q, prev_questions, threshold=0.80):
            new_q = q
            break
        else:
            print(f"[DEBUG] 생성된 질문이 기존 질문과 유사하여 재생성 시도 ({attempt+1}/3): {q}")

    # 3번 다 시도했는데도 전부 겹치면 마지막 거라도 사용
    if new_q is None:
        new_q = q


    # 스키마 준수: 새로운 키를 state에 추가하지 않음
    return {
        **state,
        "current_question": q,
        "current_answer": ""
    }



# 5) 인터뷰 피드백 보고서 --------------------



def summarize_interview(state: InterviewState) -> InterviewState:
    # 여기에 코드를 완성합니다.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 기업정보 불러오기
    company_name = state.get("company_name", "")
    company_context = get_company_context(company_name)

    question_prompt = PromptTemplate(
        input_variables=["resume_summary", "company_context", "keywords", "conversation", "evaluation"],
        template="""주어진 정보를 바탕으로 인터뷰 피드백 보고서를 생성해야합니다.
                    인터뷰 피드백 보고서는 단순히 대화를 출력하는 것이 아니라,
                    보고서를 보고 면접자가 개선할 수 있는 피드백까지를 제공해야 합니다.

                    이력서 요약: {resume_summary}
                    기업 인재상 및 핵심가치: {company_context}
                    주요 키워드: {keywords}
                    대화: {conversation}
                    평가: {evaluation}

                    위 내용을 바탕으로

                    # [인터뷰 피드백 보고서]
                    ---
                    ## [1] 경력 및 경험
                    - 답변요약
                    - 강점
                    - 약점
                    ## [2] 동기 및 커뮤니케이션
                    - 답변요약
                    - 강점
                    - 약점
                    ## [3] 논리적 사고
                    - 답변요약
                    - 강점
                    - 약점
                    ## [4] 기업의 핵심가치 및 인재상
                    - 답변요약
                    - 강점
                    - 약점
                    ## [5] 종합 피드백
                    - 피드백 1 ~ 4

                    위 마크다운 형식으로 생성합니다.

                    '[1] 경력 및 경험'
                    '[2] 동기 및 커뮤니케이션'
                    '[3] 논리적 사고'
                    '[4] 기업의 핵심가치 및 인재상'
                    위 네 가지 목록을

                    '답변요약', '강점', '약점'
                    정확히 위 세 가지 세부 사항으로 정리하여 작성하고,

                    위의 목록과 세부 사항 내용을 바탕으로한
                    마무리 '[5] 종합 피드백' 항목을 한 목록 이상
                    세 목록 이하로 나눠서 요약하여 작성해주세요.
                    """
      )

    question_chain = question_prompt | llm
    response = question_chain.invoke({
        "resume_summary": state["resume_summary"],
        "company_context": get_company_context(company_name),
        "keywords": state["resume_keywords"],
        "conversation": state["conversation"],
        "evaluation": state["evaluation"]
    })

    display(Markdown("# [면접 내용]\n---"))
    for idx, (conv, eval_data) in enumerate(zip(state["conversation"], state["evaluation"]), 1):
        display(Markdown(f"## [질문 {idx}]\n* {conv['질문']}"))
        display(Markdown(f"### [답변 {idx}]\n* {conv['답변']}"))
        display(Markdown(
    f"### [평가 {idx}]\n"
    f"* 질문과의 연관성: {eval_data['질문과의_연관성']['등급']}\n"
    f"  {eval_data['질문과의_연관성']['근거']}\n"
    f"* 답변의 구체성: {eval_data['답변의_구체성']['등급']}\n"
    f"  {eval_data['답변의_구체성']['근거']}\n"
    f"* 기업가치 부합도: {eval_data['기업가치_부합도']['등급']}\n"
    f"  {eval_data['기업가치_부합도']['근거']}"
))




    # return 코드는 제공합니다.
    return state



# ===== 3. ilm 스타일 템플릿 (독립형) =====
ilm_style_template = """
  // ilm 스타일 면접 보고서 템플릿 (독립형)
  // 원본: @preview/ilm:1.4.1 영감

  #let interview_ilm(
    title: [인터뷰 피드백 보고서],
    author: "면접관",
    date: datetime.today(),
    abstract: none,
    candidate: "",
    position: "",
    font: ("NanumGothic", "Nanum Gothic"),
    body
  ) = {
    // 페이지 설정
    set page(
      paper: "a4",
      margin: (top: 3cm, bottom: 3cm, left: 2.5cm, right: 2.5cm),
      numbering: "1",
      number-align: center,

      // 헤더 설정 (ilm 스타일)
      header: context {
        if counter(page).get().first() > 1 {
          set text(9pt, style: "italic")
          smallcaps(title)
          h(1fr)
          author
          line(length: 100%, stroke: 0.5pt)
        }
      }
    )

    // 텍스트 설정
    set text(
      font: font,
      size: 11pt,
      lang: "ko",
      fallback: true
    )

    set par(
      justify: true,
      leading: 0.65em,
      first-line-indent: 0em
    )

    // 제목 스타일
    set heading(numbering: "1.1")

    show heading.where(level: 1): it => {
      pagebreak(weak: true)
      set text(size: 20pt, weight: "bold")
      block(
        above: 1.5em,
        below: 1em,
        it
      )
    }

    show heading.where(level: 2): it => {
      set text(size: 14pt, weight: "bold")
      block(
        above: 1.2em,
        below: 0.8em,
        it
      )
    }

    show heading.where(level: 3): it => {
      set text(size: 12pt, weight: "semibold")
      block(
        above: 1em,
        below: 0.6em,
        it
      )
    }

    // 링크 스타일
    show link: it => text(fill: blue, it)

    // 리스트 스타일
    set list(indent: 1em, body-indent: 0.5em)
    set enum(indent: 1em, body-indent: 0.5em)

    // Quote 스타일
    set quote(block: true)
    show quote: set pad(left: 2em)
    show quote: it => block(
      fill: luma(240),
      width: 100%,
      inset: 10pt,
      radius: 3pt,
      it
    )

    // ===== 표지 =====
    align(center)[
      #v(2cm)

      // 제목
      #text(28pt, weight: "bold")[
        #title
      ]

      #v(1cm)

      // 저자와 날짜
      #text(14pt)[
        #author
      ]

      #v(0.5cm)

      #text(12pt)[
        #date.display("[month repr:long] [day], [year]")
      ]

      #v(1.5cm)

      // 지원자 정보
      #if candidate != "" or position != "" {
        block(
          width: 80%,
          inset: 15pt,
          fill: luma(250),
          radius: 5pt,
          [
            #if candidate != "" [
              *지원자:* #candidate \
            ]
            #if position != "" [
              *지원 직무:* #position
            ]
          ]
        )
      }

      #v(1cm)

      // Abstract
      #if abstract != none {
        block(
          width: 85%,
          inset: 20pt,
          stroke: (left: 3pt + blue),
          [
            #text(12pt, weight: "bold")[요약]
            #v(0.5em)
            #set par(justify: true)
            #abstract
          ]
        )
      }
    ]

    // 목차 (2페이지부터 시작)
    pagebreak()

    outline(
      title: [목차],
      indent: auto,
      depth: 3
    )

    // 본문
    pagebreak()

    body

    // 페이지 하단
    v(3cm)
    align(center)[
      #line(length: 30%, stroke: 0.5pt)
      #v(0.5em)
      #text(9pt, fill: gray, style: "italic")[
        본 문서는 자동으로 생성되었습니다.
      ]
    ]
}"""

# 템플릿 파일 생성
with open("interview_ilm_standalone.typ", "w", encoding="utf-8") as f:
    f.write(ilm_style_template)

print("✓ ilm 스타일 템플릿 생성 완료")

# 변환 및 생성 함수
def markdown_to_typst(text):
    text = re.sub(r'^# \[(.+)\]$', r'= \1', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'== \1', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'=== \1', text, flags=re.MULTILINE)
    text = re.sub(r'^#### (.+)$', r'==== \1', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    text = re.sub(r'^\* ', r'- ', text, flags=re.MULTILINE)
    text = re.sub(r'^---$', r'#line(length: 100%)', text, flags=re.MULTILINE)
    return text






# pdf 생성 함수
# generate_report_with_params 함수 전체 수정
def generate_report_with_params(session_state):
    """세션 상태에서 정보를 추출하여 PDF 생성"""
    if not session_state["interview_ended"]:
        return None

    state = session_state["state"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 기업정보 불러오기
    company_name = state.get("company_name", "")
    company_context = get_company_context(company_name)

    # 1. 면접 내용 마크다운 생성
    interview_content = "# [면접 내용]\n---\n\n"

    for idx, (conv, eval_data) in enumerate(zip(state["conversation"], state["evaluation"]), 1):
        interview_content += f"## [질문 {idx}]\n"
        interview_content += f"* {conv['질문']}\n\n"
        interview_content += f"### [답변 {idx}]\n"
        interview_content += f"* {conv['답변']}\n\n"
        interview_content += f"### [평가 {idx}]\n"
        interview_content += f"* 질문과의 연관성: {eval_data['질문과의_연관성']['등급']}\n  {eval_data['질문과의_연관성']['근거']}\n"
        interview_content += f"* 답변의 구체성: {eval_data['답변의_구체성']['등급']}\n  {eval_data['답변의_구체성']['근거']}\n\n"
        interview_content += f"* 기업가치 부합도: {eval_data['기업가치_부합도']['등급']}\n  {eval_data['기업가치_부합도']['근거']}\n\n"

    # 2. 종합 피드백 보고서 생성 (LLM 사용)
    question_prompt = PromptTemplate(
        input_variables=["resume_summary", "company_context", "keywords", "conversation", "evaluation"],
        template="""주어진 정보를 바탕으로 인터뷰 피드백 보고서를 생성해야합니다.

이력서 요약: {resume_summary}
기업 인재상 및 핵심가치: {company_context}
주요 키워드: {keywords}
대화: {conversation}
평가: {evaluation}

위 내용을 바탕으로

# [인터뷰 피드백 보고서]
---
## [1] 경력 및 경험
- 답변요약
- 강점
- 약점
## [2] 동기 및 커뮤니케이션
- 답변요약
- 강점
- 약점
## [3] 논리적 사고
- 답변요약
- 강점
- 약점
## [4] 기업의 핵심가치 및 인재상
- 답변요약
- 강점
- 약점
## [5] 종합 피드백
- 피드백 1 ~ 4

위 마크다운 형식으로 생성합니다.
각 항목은 반드시 '- '로 시작하는 리스트 형식으로 작성하세요."""
    )

    question_chain = question_prompt | llm
    feedback_response = question_chain.invoke({
        "resume_summary": state["resume_summary"],
        "keywords": state["resume_keywords"],
        "conversation": state["conversation"],
        "evaluation": state["evaluation"],
        "company_context": get_company_context(company_name)
    })

    # 3. 전체 마크다운 결합
    full_markdown = interview_content + "\n" + feedback_response.content

    # 4. Markdown → Typst 변환
    interview_typst = markdown_to_typst(full_markdown)

    abstract = "본 보고서는 지원자의 면접에 대한 종합 평가 및 개선 방안을 제시합니다."

    # 5. Typst 문서 생성
    doc = f"""
#import "interview_ilm_standalone.typ": *

#show: interview_ilm.with(
  title: [인터뷰 피드백 보고서],
  author: "면접관",
  date: datetime.today(),
  candidate: "지원자",
  position: "면접 직무",
  abstract: [
    {abstract}
  ],
)

{interview_typst}
"""

    # 6. .typ 파일 저장
    typ_path = "report.typ"
    with open(typ_path, "w", encoding="utf-8") as f:
        f.write(doc)

    # 7. PDF 컴파일
    pdf_path = "interview_report.pdf"
    try:
        pdf_bytes = typst.compile(typ_path)

        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        print("✅ PDF 생성 완료!")
        return pdf_path
    except Exception as e:
        print(f"❌ PDF 생성 실패: {e}")
        return None

# 6) Agent --------------------
# 분기 판단 함수

def route_next(state: InterviewState) -> Literal["generate", "summarize"]:
    if state["next_step"] == "end":
        return "summarize"
    else:
        return "generate"

# 그래프 정의 시작
workflow = StateGraph(InterviewState)

# 노드 추가
workflow.add_node("evaluate", evaluate_answer)

# 답변에 대한 평가 고도화
workflow.add_node("reflect", reflect_evaluation)  # 새로 추가
workflow.add_node("re_evaluate", re_evaluate_answer)  # 새로 추가

workflow.add_node("decide", decide_next_step)
workflow.add_node("generate", generate_question)
workflow.add_node("summarize", summarize_interview)

# 노드 연결
workflow.set_entry_point("evaluate")


# 답변에 대한 평가 고도화
workflow.add_edge("evaluate", "reflect")  # 평가 후 reflection
workflow.add_conditional_edges("reflect",
    route_after_reflection,
    {
        "re_evaluate": "re_evaluate",  # 재평가 필요
        "decide": "decide"  # 정상 → 다음 단계
    }
)
workflow.add_edge("re_evaluate", "decide")  # 재평가 후 다음 단계

workflow.add_conditional_edges(
    "decide",
    route_next,
    {
        "generate": "generate",
        "summarize": "summarize"
    }
)
workflow.add_edge("generate", END)
workflow.add_edge("summarize", END)

# 컴파일
graph = workflow.compile()


#종합 평가 등급 출력
def summarize_final_result(state: InterviewState, verbose: bool = True) -> InterviewState:
    """
    모든 면접 종료 후 전체 평균 및 최종 등급(A/B/C)과 100점 환산 점수를 계산.
    - 회차별 평균점수(0~3)를 100점 환산: round(avg * (100/3), 1)
    - 최종 등급 기준:
        A: 평균 >= 2.7
        B: 평균 >= 2.0
        C: 그 외
    """
    evaluations = state.get("evaluation", [])
    if not evaluations:
        if verbose:
            print("[DEBUG] 평가 데이터가 없습니다.")
        return state

    # 회차별 평균점수들의 평균
    per_round_avgs = [e.get("종합평가", {}).get("평균점수", 0.0) for e in evaluations]
    overall_avg = sum(per_round_avgs) / len(per_round_avgs)

    # 등급 및 100점 환산
    if overall_avg >= 2.7:
        final_grade = "A (우수)"
    elif overall_avg >= 2.0:
        final_grade = "B (보통)"
    else:
        final_grade = "C (미흡)"

    percent_score = round(overall_avg * (100.0 / 3.0), 1)

    result = {
        "평균점수_0to3": round(overall_avg, 2),
        "환산점수_0to100": percent_score,
        "최종등급_ABC": final_grade,
        "총면접회수": len(evaluations),
    }
    state["final_result"] = result

    if verbose:
        print("="*39)
        print("[최종 인터뷰 결과]")
        print(f"총 {len(evaluations)}회 면접 평균점수: {overall_avg:.2f} / 3.0")
        print(f"환산 점수: {percent_score:.1f} / 100")
        print(f"최종 등급: {final_grade}")
        print("="*39)

    return state


#-------------------------------------------------------------------

########### 다음 코드는 제공되는 gradio 코드 입니다.################

'''

# 세션 상태 초기화 함수
def initialize_state():
    return {
      "state": None,
      "interview_started": False,
      "interview_ended": False,
      "chat_history": []
    }

# 파일 업로드 후 인터뷰 초기화
def upload_and_initialize(file_obj, company_name, session_state):
    if file_obj is None:
        return session_state, "파일을 업로드해주세요."

    # Gradio는 file_obj.name 이 파일 경로야
    file_path = file_obj.name

    # 인터뷰 사전 처리
    state = preProcessing_Interview(file_path)
    state["company_name"] = company_name
    session_state["state"] = state
    session_state["interview_started"] = True

    # 첫 질문 저장
    first_question = state["current_question"]
    session_state["chat_history"].append(["🤖 AI 면접관", first_question])

    return session_state, session_state["chat_history"]

# 답변 처리 및 다음 질문 생성

def chat_interview(user_input, session_state):
    if not session_state["interview_started"]:
        yield session_state, "먼저 이력서를 업로드하고 인터뷰를 시작하세요.", gr.update(value="")
        return

    # (1) 사용자의 답변 즉시 표시
    session_state["chat_history"].append(["🙋‍♂️ 지원자", user_input])
    yield session_state, session_state["chat_history"], gr.update(value="")

    # (2) AI가 생각 중임을 바로 표시
    session_state["chat_history"].append(["🤖 AI 면접관", "💭 답변을 분석 중입니다... 잠시만 기다려주세요."])
    yield session_state, session_state["chat_history"], gr.update(value="")

    # (3) 백엔드 연산 (LLM 호출 등)
    session_state["state"] = update_current_answer(session_state["state"], user_input)
    session_state["state"] = graph.invoke(session_state["state"])

    # (4) 인터뷰 종료 여부 판단
    if session_state["state"]["next_step"] == "end":
        session_state["interview_ended"] = True
        final_summary = "✅ 인터뷰가 종료되었습니다!\n\n"

        for i, turn in enumerate(session_state["state"]["conversation"]):
            final_summary += f"\n**[질문 {i+1}]** {turn['질문']}\n**[답변 {i+1}]** {turn['답변']}\n"
            if i < len(session_state["state"]["evaluation"]):
                eval_result = session_state["state"]["evaluation"][i]
                final_summary += f"_평가 - 질문 연관성: {eval_result['질문과의_연관성']['등급']}, 답변 구체성: {eval_result['답변의_구체성']['등급']}, , 기업가치 부합도: {eval_result['기업가치_부합도']['등급']}_\n"

        # 임시 메시지를 최종 내용으로 교체
        session_state["chat_history"][-1] = ["🤖 AI 면접관", final_summary]
        yield session_state, session_state["chat_history"], gr.update(value="")
        return

    # (5) 다음 질문 생성 완료 → 교체 표시
    next_question = session_state["state"]["current_question"]
    session_state["chat_history"][-1] = ["🤖 AI 면접관", next_question]
    yield session_state, session_state["chat_history"], gr.update(value="")

with gr.Blocks(
    css="""
        body {
            background-color: #f8fafc;
        }
        .gradio-container {
            font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
        }

        /* ===== 공통 카드 ===== */
        .main-card, .upload-box, .chat-box, .download-box {
            background: #ffffff;
            border-radius: 14px;
            border: 2px solid #6b7280;
            box-shadow: 0 3px 10px rgba(0,0,0,0.06);
            padding: 24px;
            margin: 15px auto;
        }

        /* ===== 안내 박스 ===== */
        #usage-guide {
            background: #f3f4f6;
            border: 2px solid #6b7280;
            border-radius: 10px;
            padding: 16px 18px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            color: #333;
            line-height: 1.6em;
        }

        /* ===== 최신 Gradio Chatbot 구조 대응 ===== */
        .chat-message {
            display: flex !important;
            margin: 10px 0 !important;
            padding: 0 10px !important;
        }

        /* 🤖 AI 면접관 (왼쪽 회색 말풍선) */
        .chat-message[data-testid="bot"] {
            justify-content: flex-start !important;
        }
        .chat-message[data-testid="bot"] .message-content {
            background-color: #e5e7eb !important;
            color: #111827 !important;
            border-radius: 14px 14px 14px 0 !important;
            padding: 10px 14px !important;
            max-width: 70%;
            line-height: 1.5em;
        }

        /* 🙋‍♂️ 지원자 (오른쪽 파란 말풍선) */
        .chat-message[data-testid="user"] {
            justify-content: flex-end !important;
        }
        .chat-message[data-testid="user"] .message-content {
            background-color: #2563eb !important;
            color: #ffffff !important;
            border-radius: 14px 14px 0 14px !important;
            padding: 10px 14px !important;
            max-width: 70%;
            line-height: 1.5em;
            text-align: right;
        }

        /* 입력창 스타일 */
        textarea {
            border: 2px solid #6b7280 !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
            font-size: 1em !important;
        }

        /* 버튼 */
        .gr-button {
            border-radius: 10px !important;
            font-weight: 600;
            transition: all 0.15s ease;
        }
        .gr-button:hover {
            transform: scale(1.03);
            background-color: #111827 !important;
            color: white !important;
        }
    """
) as demo:

    session_state = gr.State(initialize_state())

    # 헤더
    with gr.Column(elem_classes=["main-card"]):
        gr.Markdown("""
            <div style='text-align:center;'>
                <h1>🏢 AI 기업 맞춤 면접관</h1>
                <p style='font-size:1.05em; color:#555;'>
                원하는 기업을 선택하고, 이력서를 업로드하여 맞춤형 면접을 시작하세요.
                </p>
            </div>
        """)

    # Step 1️⃣ 기업 선택 및 업로드
    with gr.Row(elem_classes=["upload-box"]):
        with gr.Column(scale=2):
            gr.Markdown("## 1️⃣ Step 1. 기업 선택 및 이력서 업로드")
            company_dropdown = gr.Dropdown(
                choices=["CJ", "현대자동차", "삼성전자", "LG", "카카오"],
                label="지원 기업 선택",
                value="삼성전자"
            )
            file_input = gr.File(label="이력서 업로드 (PDF / DOCX)", file_types=[".pdf", ".docx"])

        with gr.Column(scale=1, min_width=230):
            gr.Markdown("""
            ### 🧭 간단 사용법
            <div id="usage-guide">

            1️⃣ **기업 선택**
            → 원하는 기업을 선택하세요.
            (면접 질문 톤이 기업별로 달라집니다.)

            2️⃣ **이력서 업로드**
            → PDF 또는 DOCX 파일을 올리면
            AI가 내용을 분석해 첫 질문을 생성합니다.

            3️⃣ **답변 입력**
            → 질문에 자연스럽게 답변하세요.
            AI가 즉시 평가 및 다음 질문을 제공합니다.

            4️⃣ **보고서 다운로드**
            → 인터뷰 종료 후 ‘보고서 다운로드’로
            개인 맞춤 피드백 PDF를 받아보세요.
            </div>
            """)
            upload_btn = gr.Button("🚀 인터뷰 시작", variant="primary")

    # Step 2️⃣ 면접 진행
    with gr.Column(elem_classes=["chat-box"]):
        gr.Markdown("## 2️⃣ Step 2. AI 면접 진행")
        chatbot = gr.Chatbot(label="AI 면접 진행", height=450, bubble_full_width=False)
        user_input = gr.Textbox(show_label=False, placeholder="답변을 입력하세요...")

    # Step 3️⃣ 보고서 다운로드
    with gr.Row(elem_classes=["download-box"]):
        with gr.Column(scale=2):
            gr.Markdown("## 3️⃣ Step 3. 피드백 보고서 다운로드")
        with gr.Column(scale=1, min_width=180):
            download_btn = gr.Button("📥 보고서 다운로드")
            report_file = gr.File(label="면접 피드백 보고서", visible=True)

    # 이벤트 연결
    upload_btn.click(upload_and_initialize, inputs=[file_input, company_dropdown, session_state], outputs=[session_state, chatbot])
    user_input.submit(chat_interview, inputs=[user_input, session_state], outputs=[session_state, chatbot])
    download_btn.click(generate_report_with_params, inputs=[session_state], outputs=[report_file])
    user_input.submit(lambda: "", None, user_input)

demo.launch(share=True)

'''


# 세션 상태 초기화 함수
def initialize_state():
    return {
      "state": None,
      "interview_started": False,
      "interview_ended": False,
      "chat_history": []
    }

# 파일 업로드 후 인터뷰 초기화
def upload_and_initialize(file_obj, company_name, session_state):
    if file_obj is None:
        return session_state, [{"role": "assistant", "content": "파일을 업로드해주세요."}]

    # Gradio는 file_obj.name 이 파일 경로야
    file_path = file_obj.name

    # 인터뷰 사전 처리
    state = preProcessing_Interview(file_path)
    state["company_name"] = company_name
    session_state["state"] = state
    session_state["interview_started"] = True

    # 첫 질문 저장
    first_question = state["current_question"]
    session_state["chat_history"].append(append_message("assistant", first_question))

    return session_state, session_state["chat_history"]

# 메세지 형식
def append_message(role, text):
    # 역할별 색상
    styles = {
        "assistant": {
            "name": "🤖 AI 면접관",
            "bg": "#2563eb",
            "color": "white"
        },
        "user": {
            "name": "🙋‍♂️ 지원자",
            "bg": "#E5E7EB",
            "color": "#111827"
        }
    }

    s = styles[role]

    # 라벨 (pill)
    label_html = f"""
    <div style='margin-bottom:4px; font-weight:600;'>
      <span style='background:{s["bg"]}; color:{s["color"]};
      padding:4px 10px; border-radius:20px; font-size:0.85em;'>
        {s["name"]}
      </span>
    </div>
    """

    # 메시지 정렬 — 지원자는 오른쪽 정렬
    align = "right" if role == "user" else "left"

    return {
        "role": role,
        "content": f"<div style='text-align:{align}; white-space:normal; word-break:break-word;'>{label_html}<div>{text}</div></div>"
    }

# 답변 처리 및 다음 질문 생성

def chat_interview(user_input, session_state):
    if not session_state["interview_started"]:
        yield session_state, [{"role": "assistant", "content": "먼저 이력서를 업로드하고 인터뷰를 시작하세요."}], gr.update(value="")
        return

    # (1) 사용자의 답변 즉시 표시
    session_state["chat_history"].append(append_message("user", user_input))
    yield session_state, session_state["chat_history"], gr.update(value="")

    # (2) AI가 생각 중임을 바로 표시
    session_state["chat_history"].append({"role": "assistant", "content": "💭 답변을 분석 중입니다... 잠시만 기다려주세요."})
    yield session_state, session_state["chat_history"], gr.update(value="")

    # (3) 백엔드 연산 (LLM 호출 등)
    session_state["state"] = update_current_answer(session_state["state"], user_input)
    session_state["state"] = graph.invoke(session_state["state"])

    # (4) 인터뷰 종료 여부 판단
    if session_state["state"]["next_step"] == "end":
        session_state["interview_ended"] = True
        final_summary = "✅ 인터뷰가 종료되었습니다!\n\n"

        for i, turn in enumerate(session_state["state"]["conversation"]):
            final_summary += f"\n**[질문 {i+1}]** {turn['질문']}\n**[답변 {i+1}]** {turn['답변']}\n"
            if i < len(session_state["state"]["evaluation"]):
                eval_result = session_state["state"]["evaluation"][i]
                final_summary += f"_평가 - 질문 연관성: {eval_result['질문과의_연관성']['등급']}, 답변 구체성: {eval_result['답변의_구체성']['등급']}, , 기업가치 부합도: {eval_result['기업가치_부합도']['등급']}_\n"

        # 임시 메시지를 최종 내용으로 교체
        session_state["chat_history"][-1] = append_message("assistant", final_summary)
        yield session_state, session_state["chat_history"], gr.update(value="")
        return

    # (5) 다음 질문 생성 완료 → 교체 표시
    next_question = session_state["state"]["current_question"]
    session_state["chat_history"][-1] = append_message("assistant", next_question)
    yield session_state, session_state["chat_history"], gr.update(value="")

# Chatbot 전용 테마 정의
chatbot_theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="gray",
).set(
    body_background_fill="#f8fafc",
    block_border_width="2px",
    block_border_color="#6b7280",
)

with gr.Blocks(
    css="""
        body {
            background-color: #f8fafc;
        }
        .gradio-container {
            font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
        }

        /* ===== 공통 카드 ===== */
        .main-card, .upload-box, .chat-box, .download-box {
            background: #ffffff;
            border-radius: 14px;
            border: 2px solid #6b7280;
            box-shadow: 0 3px 10px rgba(0,0,0,0.06);
            padding: 24px;
            margin: 15px auto;
        }

        /* 안내 박스 */
        #usage-guide {
            background: #f3f4f6;
            border: 2px solid #6b7280;
            border-radius: 10px;
            padding: 16px 18px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            color: #333;
            line-height: 1.6em;
        }

        /* Chatbot 말풍선 스타일 (전역) */
        .message-row.user .bubble {
            background-color: #2563eb !important;
            color: #fff !important;
            border-radius: 14px 14px 0 14px !important;
            text-align: right;
        }

        .message-row.assistant .bubble {
            background-color: #e5e7eb !important;
            color: #111827 !important;
            border-radius: 14px 14px 14px 0 !important;
        }

        /* 라벨 (pill 스타일) */
        .chat-label {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .label-user {
            background-color: #2563eb;
            color: white;
        }
        .label-assistant {
            background-color: #e5e7eb;
            color: #111827;
        }

        /* 입력창 / 버튼 */
        textarea {
            border: 2px solid #6b7280 !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
        }
        .gr-button {
            border-radius: 10px !important;
            font-weight: 600;
            transition: all 0.15s ease;
        }
        .gr-button:hover {
            transform: scale(1.03);
            background-color: #111827 !important;
            color: white !important;
        }
    """
) as demo:

    session_state = gr.State(initialize_state())

    # 헤더
    with gr.Column(elem_classes=["main-card"]):
        gr.Markdown("""
            <div style='text-align:center;'>
                <h1>🏢 AI 기업 맞춤 면접관</h1>
                <p style='font-size:1.05em; color:#555;'>
                원하는 기업을 선택하고, 이력서를 업로드하여 맞춤형 면접을 시작하세요.
                </p>
            </div>
        """)

    # Step 1️⃣ 기업 선택 및 업로드
    with gr.Row(elem_classes=["upload-box"]):
        with gr.Column(scale=2):
            gr.Markdown("## 1️⃣ Step 1. 기업 선택 및 이력서 업로드")
            company_dropdown = gr.Dropdown(
                choices=["CJ", "현대자동차", "삼성전자", "LG", "카카오"],
                label="지원 기업 선택",
                value="삼성전자"
            )
            file_input = gr.File(label="이력서 업로드 (PDF / DOCX)", file_types=[".pdf", ".docx"])

        with gr.Column(scale=1, min_width=230):
            gr.Markdown("""
            ### 🧭 간단 사용법
            <div id="usage-guide">
            1️⃣ **기업 선택**
            → 원하는 기업을 선택하세요.
            (면접 질문 톤이 기업별로 달라집니다.)

            2️⃣ **이력서 업로드**
            → PDF 또는 DOCX 파일을 올리면
            AI가 내용을 분석해 첫 질문을 생성합니다.

            3️⃣ **답변 입력**
            → 질문에 자연스럽게 답변하세요.
            AI가 즉시 평가 및 다음 질문을 제공합니다.

            4️⃣ **보고서 다운로드**
            → 인터뷰 종료 후 ‘보고서 다운로드’로
            개인 맞춤 피드백 PDF를 받아보세요.
            </div>
            """)
            upload_btn = gr.Button("🚀 인터뷰 시작", variant="primary")

    # Step 2️⃣ 면접 진행
    with gr.Column(elem_classes=["chat-box"]):
        gr.Markdown("## 2️⃣ Step 2. AI 면접 진행")
        chatbot = gr.Chatbot(label="AI 면접 진행", height=450, bubble_full_width=False, type="messages")
        user_input = gr.Textbox(show_label=False, placeholder="답변을 입력하세요...")

    # Step 3️⃣ 보고서 다운로드
    with gr.Row(elem_classes=["download-box"]):
        with gr.Column(scale=2):
            gr.Markdown("## 3️⃣ Step 3. 피드백 보고서 다운로드")
        with gr.Column(scale=1, min_width=180):
            download_btn = gr.Button("📥 보고서 다운로드")
            report_file = gr.File(label="면접 피드백 보고서", visible=True)

    # 이벤트 연결
    upload_btn.click(upload_and_initialize, inputs=[file_input, company_dropdown, session_state], outputs=[session_state, chatbot])
    user_input.submit(chat_interview, inputs=[user_input, session_state], outputs=[session_state, chatbot])
    download_btn.click(generate_report_with_params, inputs=[session_state], outputs=[report_file])
    user_input.submit(lambda: "", None, user_input)

demo.launch(share=True)
