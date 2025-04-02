import streamlit as st
import boto3
import json
import base64
from PIL import Image
import io
import os
import pyperclip
import re

def get_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name='us-east-1'
    )

def process_image_for_bedrock(image):
    # RGBA 이미지를 RGB로 변환
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # 이미지를 바이트로 변환
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    return img_bytes

def invoke_model(client, prompt, image_b64=None, model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0"):
    try:
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name='us-east-1'
        )

        system_prompt = f"""[분석 요청]
당신은 기사를 작성하는 전문가입니다."""

        # 기본 텍스트 콘텐츠
        content = [{"text": prompt}]
        
        # 이미지가 있는 경우 추가
        if image_b64:
            content.append({
                "image": {
                    "format": "jpeg",
                    "source": {
                        "bytes": image_b64
                    }
                }
            })

        # Call Bedrock
        response = bedrock_runtime.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{
                "role": "user",
                "content": content
            }],
            inferenceConfig={
                "maxTokens": 3000,
                "temperature": 0.3,
            }
        )

        try:
            response_text = response['output']['content'][0]['text']
        except:
            try:
                response_text = response['output']['content']['text']
            except:
                try:
                    response_text = response['output']['message']['content'][0]['text']
                except:
                    try:
                        response_text = response['output']['message']['text']
                    except:
                        response_text = str(response)

        return response_text
            
    except Exception as e:
        st.error(f"모델 호출 중 오류 발생: {str(e)}")
        print(f"상세 오류: {str(e)}")
        return None

def check_facts(text, image_b64=None):
    client = get_bedrock_client()
    prompt = f"""
    다음 텍스트의 사실 관계를 검증하고 신뢰할 수 있는 정보와 검증이 필요한 정보를 구분해서 분석해주세요:
    
    {text}
    
    [분석 형식]
    1. 신뢰할 수 있는 정보:
    - (정보 1)
    - (정보 2)
    
    2. 검증이 필요한 정보:
    - (정보 1): (검증 필요 이유)
    - (정보 2): (검증 필요 이유)
    """
    return invoke_model(client, prompt, image_b64)

def analyze_data(text, image_b64=None):
    client = get_bedrock_client()
    prompt = f"""
    다음 텍스트에 포함된 데이터를 분석하고 주요 인사이트를 도출해주세요:
    
    {text}
    
    [분석 형식]
    1. 주요 데이터 포인트:
    - (데이터 1)
    - (데이터 2)
    
    2. 인사이트:
    - (인사이트 1)
    - (인사이트 2)
    
    3. 추천 사항:
    - (추천 1)
    - (추천 2)
    """
    return invoke_model(client, prompt, image_b64)

def check_grammar(text, image_b64=None):
    client = get_bedrock_client()
    prompt = f"""
    다음 텍스트의 맞춤법과 문법을 검사하고 수정 사항을 제안해주세요:
    
    {text}
    
    [분석 형식]
    1. 맞춤법 오류:
    - (오류 1) → (수정안)
    - (오류 2) → (수정안)
    
    2. 문법 개선사항:
    - (개선 1)
    - (개선 2)
    
    3. 수정된 전체 텍스트:
    (수정된 텍스트)
    """
    return invoke_model(client, prompt, image_b64)

def generate_seo_title(text, image_b64=None):
    client = get_bedrock_client()
    prompt = f"""
    다음 텍스트를 바탕으로 SEO에 최적화된 제목을 5개 생성해주세요:
    
    {text}
    
    [생성 형식]
    1. (제목 1) - (SEO 최적화 포인트)
    2. (제목 2) - (SEO 최적화 포인트)
    3. (제목 3) - (SEO 최적화 포인트)
    4. (제목 4) - (SEO 최적화 포인트)
    5. (제목 5) - (SEO 최적화 포인트)
    """
    return invoke_model(client, prompt, image_b64)

def rewrite_text(text, style, use_emoji=False, image_b64=None):
    client = get_bedrock_client()
    emoji_instruction = "이모티콘을 적절히 사용하여 " if use_emoji else ""
    style_instructions = {
        "권위있는 기사체": """
        - 객관적이고 공식적인 톤 유지
        - 정확한 사실과 데이터 중심
        - 전문가적인 분석과 통찰 포함
        - 격식있는 어휘 사용
        """,
        "르포 기사체": """
        - 현장감 있는 묘사
        - 구체적인 디테일 포함
        - 인터뷰와 증언 활용
        - 생생한 스토리텔링
        """,
        "세련된 뉴스레터체": """
        - 친근하고 대화체적인 톤
        - 핵심 포인트 강조
        - 간결하고 명확한 문장
        - 독자와 공감대 형성
        """,
        "AXIOS 기사체": """
        - 핵심 정보 먼저 제시
        - 짧고 명확한 문단
        - 불렛 포인트 활용
        - Why it matters 섹션 포함
        """
    }

    prompt = f"""
    다음 텍스트를 {emoji_instruction}{style} 스타일로 다시 작성해주세요:
    
    [스타일 가이드라인]
    {style_instructions[style]}

    [원문]
    {text}
    """
    return invoke_model(client, prompt, image_b64)

def main():
    st.set_page_config(page_title="AI Writing Assistant", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .stTabs {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
        }
        .stButton button {
            background-color: #4C7BF4;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 2rem;
        }
        .word-counter {
            color: #6c757d;
            text-align: right;
            font-size: 0.9em;
        }
        .custom-tabs {
            display: flex;
            justify-content: flex-start;
            gap: 10px;
            margin-bottom: 20px;
        }
        .custom-tab {
            padding: 10px 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
            cursor: pointer;
        }
        .custom-tab.active {
            background-color: #4C7BF4;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None

    # 상단 탭
    tabs = ["기사 작성", "팩트 체크", "데이터 분석", "맞춤법 교정"]
    selected_tab = st.radio("메뉴", tabs, horizontal=True, label_visibility="collapsed")

    if selected_tab == "팩트 체크":
    # 세션 상태 초기화
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("pages/1_fact_check.py")
        return
    elif selected_tab == "데이터 분석":
        # 세션 상태 초기화
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("pages/2_data_analysis.py")
        return
    elif selected_tab == "맞춤법 교정":
        # 세션 상태 초기화
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("pages/3_grammar_check.py")
        return
    # 서브 메뉴 컨테이너
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            model = st.selectbox("모델", ["Nova-Pro"], label_visibility="collapsed")
        
        with col2:
            style = st.selectbox("스타일", ["권위있는 기사체", "르포 기사체", "세련된 뉴스레터체", "AXIOS 기사체"], label_visibility="collapsed")
            
        with col3:
            tone = st.selectbox("어조", ["경어체", "반말체", "중립적"], label_visibility="collapsed")
            
        with col4:
            audience = st.selectbox("대상", ["일반 대중", "전문가", "청소년"], label_visibility="collapsed")
            
        with col5:
            word_limit = st.selectbox("글자수", ["1000자", "2000자", "3000자"], label_visibility="collapsed")

    # 메인 입력 영역
    text_input = st.text_area(
        "아래 prompt를 기반으로 기사를 작성해라.",
        value=st.session_state.current_text,
        height=300,
        key="text_input"
    )

    # 이미지 업로드 영역
    uploaded_image = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="업로드된 이미지")
        # 이미지를 base64로 변환하여 세션 상태에 저장
        st.session_state.current_image = process_image_for_bedrock(image)

    # 하단 기능 버튼들과 카운터
    col_buttons = st.columns([1, 6, 1, 1, 1])
    
    with col_buttons[0]:
        use_emoji = st.selectbox("이모티콘 사용", ["이모티콘 사용 안함", "이모티콘 사용"], label_visibility="collapsed") == "이모티콘 사용"
    
    with col_buttons[1]:
        if st.button("작성하기", use_container_width=True):
            if text_input.strip():
                with st.spinner('처리 중...'):
                    result = None
                    image_b64 = st.session_state.get('current_image')
                    if selected_tab == "기사 작성":
                        result = rewrite_text(text_input, style, use_emoji, image_b64)
                    elif selected_tab == "데이터 분석":
                        result = analyze_data(text_input, image_b64)
                    elif selected_tab == "맞춤법 교정":
                        result = check_grammar(text_input, image_b64)
                    elif selected_tab == "SEO 제목":
                        result = generate_seo_title(text_input, image_b64)
                    
                    if result:
                        st.session_state.history.append(text_input)
                        st.session_state.current_text = result
                        st.markdown("### 결과")
                        st.write(result)
    
    with col_buttons[2]:
        if st.button("관련 변경", use_container_width=True):
            if text_input.strip():
                with st.spinner('처리 중...'):
                    client = get_bedrock_client()
                    prompt = f"다음 텍스트와 관련된 다른 주제나 관점으로 변경해서 작성해주세요:\n\n{text_input}"
                    image_b64 = st.session_state.get('current_image')
                    response = invoke_model(client, prompt, image_b64)
                    if response:
                        st.session_state.history.append(text_input)
                        st.session_state.current_text = response
                        st.markdown("### 결과")
                        st.write(response)
    
    with col_buttons[3]:
        if st.button("재작성", use_container_width=True):
            if text_input.strip():
                with st.spinner('처리 중...'):
                    client = get_bedrock_client()
                    prompt = f"다음 텍스트를 완전히 새로운 방식으로 재작성해주세요:\n\n{text_input}"
                    image_b64 = st.session_state.get('current_image')
                    response = invoke_model(client, prompt, image_b64)
                    if response:
                        st.session_state.history.append(text_input)
                        st.session_state.result = response
                        st.session_state.current_text = ""
                        st.rerun()
    
    with col_buttons[4]:
        if st.button("복사", use_container_width=True):
            try:
                pyperclip.copy(text_input)
                st.success("클립보드에 복사되었습니다!")
            except Exception as e:
                st.error(f"복사 중 오류가 발생했습니다: {str(e)}")

    # 글자 수 카운터
    current_chars = len(text_input)
    st.markdown(f'<p class="word-counter">{current_chars}자/3,000자</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
