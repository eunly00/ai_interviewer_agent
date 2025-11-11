# AI 면접 에이전트 (AI Interview Agent)
<img width="897" height="648" alt="image" src="https://github.com/user-attachments/assets/5c017aea-8894-437c-bc7d-555111eb99f7" />
## 프로젝트 소개
이 프로젝트는  **AI 기반 면접 연습 에이전트**입니다.  
이력서(PDF/DOCX)를 업로드하면, ChatGPT와 LangChain을 활용해 맞춤형 면접 질문을 생성하고,  
답변에 대해 **연관성, 구체성, 기업가치 부합도** 등을 평가하여 피드백을 제공합니다.

## 주요 기능
- [x] 이력서(PDF/DOCX) 텍스트 추출 및 요약
- [x] 이력서 기반 맞춤형 면접 질문 생성
- [x] 답변에 대한 자동 평가(연관성 / 구체성 / 기업가치 부합도)
- [x] 인터뷰 피드백 보고서(PDF) 자동 생성


<img width="482" height="545" alt="image" src="https://github.com/user-attachments/assets/f9450c05-9f36-409a-a0bb-1e673150e439" />


##  기술 스택
- **Language**: Python
- **LLM**: OpenAI GPT-4.1, GPT-4o-mini
- **Framework**: LangChain, LangGraph
- **UI**: Gradio
- **PDF 생성**: Typst

##  실행 방법
```bash
pip install -r requirements.txt
python app.py




