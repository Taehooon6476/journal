import streamlit as st
import boto3
import json
from PIL import Image
import io

def get_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name='us-east-1'
    )

def process_image_for_bedrock(image):
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    return img_bytes

def invoke_model(client, prompt, image_b64=None, model_id="us.anthropic.claude-3-sonnet-20240229-v1:0"):
    try:
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name='us-east-1'
        )

        system_prompt = f"""[분석 요청]
당신은 한국어 맞춤법과 문법 전문가입니다. 주어진 텍스트의 맞춤법과 문법을 철저히 검토하고 개선점을 제안해주세요."""

        content = [{"text": prompt}]
        
        if image_b64:
            content.append({
                "image": {
                    "format": "jpeg",
                    "source": {
                        "bytes": image_b64
                    }
                }
            })

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

def check_grammar(text, image_b64=None):
    client = get_bedrock_client()
    prompt = f"""
    다음 텍스트의 맞춤법과 문법을 검사하고 상세한 분석과 수정 사항을 제안해주세요:
    
    [원문]
    {text}
    
    [분석 요청사항]
    1. 맞춤법 오류:
    - 오류 단어 → 올바른 표현
    - 오류 이유 설명
    
    2. 문법적 개선사항:
    - 어색한 문장 구조
    - 조사 사용의 적절성
    - 문장 호응 관계
    
    3. 문체 및 스타일:
    - 일관성 있는 어조 사용
    - 적절한 존댓말/반말 사용
    - 전문용어 사용의 적절성
    
    4. 수정된 전체 텍스트:
    (모든 수정사항이 반영된 최종본)
    
    5. 추가 제안사항:
    - 가독성 향상을 위한 제안
    - 문장 구조 개선 제안
    """
    return invoke_model(client, prompt, image_b64)

def main():
    st.set_page_config(page_title="맞춤법 교정", layout="wide")
    
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
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'grammar_result' not in st.session_state:
        st.session_state.grammar_result = None

    # 상단 탭 (현재 탭 활성화)
    tabs = ["기사 작성", "팩트 체크", "데이터 분석", "맞춤법 교정", "SEO 제목"]
    selected_tab = st.radio("메뉴", tabs, index=3, horizontal=True, label_visibility="collapsed")
    
    # 다른 탭 클릭 시 해당 페이지로 이동
    if selected_tab == "기사 작성":
        st.switch_page("app")
    elif selected_tab == "팩트 체크":
        st.switch_page("fact_check")
    elif selected_tab == "데이터 분석":
        st.switch_page("data_analysis")

    # 서브 메뉴
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("모델", ["Nova_pro"], label_visibility="collapsed")
    with col2:
        model = st.selectbox("언어", ["한국어"], label_visibility="collapsed")

    # 메인 콘텐츠 영역
    col_left, col_right = st.columns(2)
    
    with col_left:
        text_input = st.text_area(
            "맞춤법을 검사할 텍스트를 입력하세요",
            value=st.session_state.current_text,
            height=400,
            key="text_input"
        )
        
        if st.button("검사하기", use_container_width=True):
            if text_input.strip():
                with st.spinner('처리 중...'):
                    result = check_grammar(text_input)
                    if result:
                        st.session_state.grammar_result = result
                        st.rerun()
    
    with col_right:
        if st.session_state.grammar_result:
            st.write(st.session_state.grammar_result)

    # 글자 수 카운터
    current_chars = len(text_input)
    st.markdown(f'<p class="word-counter">{current_chars}자/3,000자</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
